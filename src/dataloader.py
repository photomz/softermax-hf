import itertools
import os
from dataclasses import dataclass, field
from typing import Dict, Literal, Optional
from functools import cached_property

import torch
from datasets import IterableDatasetDict, concatenate_datasets, load_dataset
from torch.utils.data import DataLoader
from transformers import (
    PreTrainedTokenizerBase,
    DataCollatorForLanguageModeling,
    DataCollatorWithPadding,
    DefaultDataCollator,
)


@dataclass
class BooksCorpusAndWiki:
    tokenizer: PreTrainedTokenizerBase

    # stream shuffled dataset by batches of buffer_size
    buffer_size: int = 30000
    max_seq_length: int = 128
    batch_size: Dict[str, int] = field(default_factory=lambda: {"train": 32, "validation": 32})

    # % tokens to mask for Encoders. Set to 0% to disable for Decoders.
    mlm_probability: float = 0.15
    # random seed for shuffling and masking.
    seed: int = 42
    # HuggingFace iterable dataset dict.
    datasets: Optional[IterableDatasetDict] = None

    # no. cpu threads. Defaults to # cores - 2.
    # these are used to determine how many threads will process the streaming
    # dataset concurrently with model training
    num_workers: int = os.cpu_count() - 2

    def setup(self, suffix=""):
        """Streams, shuffles, tokenizes, and groups the datasets.

        Args:
            suffix (str, optional): % of datasplit. Defaults to 100%. Format: [:n%]
        """

        # stream the dataset as it is too large to download
        bookcorpus = load_dataset("bookcorpus", split=f"train{suffix}", streaming=True)
        wiki_train = load_dataset("wiki40b", "en", split=f"train{suffix}", streaming=True)
        wiki_val = load_dataset("wiki40b", "en", split=f"validation{suffix}", streaming=True)

        # keep text column to concat with bookcorpus
        wiki_train = wiki_train.remove_columns([col for col in wiki_train.column_names if col != "text"])
        wiki_val = wiki_val.remove_columns([col for col in wiki_val.column_names if col != "text"])

        assert bookcorpus.features.type == wiki_train.features.type

        # dataset transform steps
        self.datasets = (
            IterableDatasetDict({"train": concatenate_datasets([bookcorpus, wiki_train]), "validation": wiki_val})
            .shuffle(seed=self.seed, buffer_size=self.buffer_size)
            .map(self.encode, batched=True, remove_columns=wiki_val.column_names)
            .map(self.group, batched=True)
            .with_format("torch")
        )

        assert isinstance(self.datasets["train"], torch.utils.data.IterableDataset)
        assert isinstance(self.datasets["validation"], torch.utils.data.IterableDataset)

    @cached_property
    def training_args(self):
        """
        Dataset-specific **kwargs to init HuggingFace TrainingArguments
        Ref: https://huggingface.co/docs/transformers/v4.36.1/en/main_classes/trainer#transformers.TrainingArguments
        """
        return {
            "dataloader_num_workers": self.num_workers,
            "data_seed": self.seed,
        }

    @cached_property
    def trainer_params(self):
        """
        Dataset-specific **kwargs to init HF Trainer
        Ref: https://huggingface.co/docs/transformers/main_classes/trainer#transformers.Trainer
        """
        # we initialise different data collators based on causal or masked language modeling
        if self.mlm_probability:
            # in MLM we usually only have [CLS] and [SEP] tokens
            collate_fn = DataCollatorForLanguageModeling(self.tokenizer, mlm=True, mlm_probability=self.mlm_probability)
        else:
            # DataCollatorForLanguageModeling(mlm=False) requires a pad_token (which doesn't exist) to be set, but
            # our method of CLM pretraining we chunk the input (see group function) for pretraining with large amounts
            # of data, hence no [PAD] token. This means the outputs of the dataloader are all `max_seq_len` long.
            # So here we explicitly use the default data collator which doesn't include any padding, outputting samples as is.
            # The caveat here is that we have to manually create the 'labels' column, which we do at the end of .group()
            collate_fn = DefaultDataCollator(return_tensors="pt")

        return {
            "train_dataset": self.datasets["train"],
            "eval_dataset": self.datasets["validation"],
            "data_collator": collate_fn,
        }

    @cached_property
    def calibration_split(self):
        """
        Dataset-specific calibration samples for GPTQConfig, returned as list of strings (since that's the format they require)
        The calibration split is equal to the validation split. Important to be a cached_property since it's rather
        expensive to turn the whole eval dataset into a list every time the getter is called.
        """
        return list(self.datasets["validation"].take(self.quant_dataset_size))

    @property
    def quant_dataset_size(self):
        # quantization calibration dataset size
        # original GPTQ paper used 128 random 2048 token segments from the C4 dataset
        # we match the same token count calculating using max_seq_len parameter
        return int(128 * 2048 / self.max_seq_length)

    def dataloader(self, split: Literal["train", "validation"] = "train"):
        torch.manual_seed(self.seed)
        # we initialise different data collators based on causal or masked language modeling
        if self.mlm_probability:
            # in MLM we usually only have [CLS] and [SEP] tokens
            collate_fn = DataCollatorForLanguageModeling(self.tokenizer, mlm=True, mlm_probability=self.mlm_probability)
        else:
            collate_fn = DefaultDataCollator(return_tensors="pt")
        return DataLoader(
            self.datasets[split],
            batch_size=self.batch_size[split],
            num_workers=self.num_workers,
            collate_fn=collate_fn,
        )

    def group(self, example_batch):
        """
        Concatenate the dataset into chunks of max_seq_length. Don't waste tokens exceeding seqlen.
        Ref: https://huggingface.co/learn/nlp-course/chapter7/3?fw=pt#preprocessing-the-data
        """
        # Flatten batch
        concatenated_examples = {k: list(itertools.chain(*example_batch[k])) for k in example_batch.keys()}
        total_length = len(concatenated_examples[list(example_batch.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of
        # this drop, you can customize this part to your needs.
        if total_length >= self.max_seq_length:
            total_length = (total_length // self.max_seq_length) * self.max_seq_length
        # Split by chunks of max_len.
        result = {
            k: [t[i : i + self.max_seq_length] for i in range(0, total_length, self.max_seq_length)]
            for k, t in concatenated_examples.items()
        }

        # create new labels column
        # remember the model will automatically right-shift them for us when calculating loss
        result["labels"] = result["input_ids"].copy()
        return result

    def encode(self, example_batch):
        """
        Tokenizes all the text into tokens first, the chunking will be done after by .group() function
        so we don't unnecessarily drop tokens by using huggingface's built-in chunking method.
        Ref: https://huggingface.co/learn/nlp-course/chapter7/6?fw=pt#preparing-the-dataset
        """
        # tokenize the text
        features = self.tokenizer(
            example_batch["text"],
            return_special_tokens_mask=True,
        )
        return features

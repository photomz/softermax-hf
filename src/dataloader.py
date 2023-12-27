import itertools
import os
from dataclasses import dataclass, field
from typing import Dict, Literal, Optional

import torch
from datasets import IterableDatasetDict, concatenate_datasets, load_dataset
from torch.utils.data import DataLoader
from transformers import DataCollatorForLanguageModeling, PreTrainedTokenizerBase


@dataclass
class BooksCorpusAndWiki:
    tokenizer: PreTrainedTokenizerBase
    # stream shuffled dataset by batches of buffer_size
    buffer_size: int = 10000
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

    @property
    def training_args(self):
        """
        Dataset-specific **kwargs to init HuggingFace TrainingArguments
        Ref: https://huggingface.co/docs/transformers/v4.36.1/en/main_classes/trainer#transformers.TrainingArguments
        """
        return {
            "dataloader_num_workers": self.num_workers,
            "per_device_train_batch_size": self.batch_size["train"],
            "per_device_eval_batch_size": self.batch_size["validation"],
            "data_seed": self.seed,
        }

    @property
    def trainer_params(self):
        """
        Dataset-specific **kwargs to init HF Trainer
        Ref: https://huggingface.co/docs/transformers/main_classes/trainer#transformers.Trainer
        """
        collate_fn = DataCollatorForLanguageModeling(
            self.tokenizer, mlm=(self.mlm_probability != 0), mlm_probability=self.mlm_probability
        )
        return {
            "train_dataset": self.datasets["train"],
            "eval_dataset": self.datasets["validation"],
            "data_collator": collate_fn,
        }

    def dataloader(self, split: Literal["train", "validation"] = "train"):
        torch.manual_seed(self.seed)

        return DataLoader(
            self.datasets[split],
            batch_size=self.batch_size[split],
            num_workers=self.num_workers,
            collate_fn=DataCollatorForLanguageModeling(
                self.tokenizer, mlm=(self.mlm_probability != 0), mlm_probability=self.mlm_probability
            ),
        )

    def group(self, example_batch):
        """Concatenate the dataset into chunks of max_seq_length. Don't waste tokens exceeding seqlen."""
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
        return result

    def encode(self, example_batch):
        # tokenize the text
        features = self.tokenizer(
            example_batch["text"],
            max_length=self.max_seq_length,
            padding="max_length",
            return_special_tokens_mask=True,
            return_tensors="pt",
        )
        return features

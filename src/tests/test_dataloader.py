import pytest
from transformers import AutoModel, RobertaTokenizer, Trainer, TrainingArguments

from src.dataloader import BooksCorpusAndWiki


# These tests require Internet to download the dataset, or a precached dataset.
@pytest.fixture(scope="session")
def bookscorpusandwiki():
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    loader = BooksCorpusAndWiki(tokenizer)
    loader.setup()
    return loader


@pytest.fixture(scope="session")
def model():
    return AutoModel.from_pretrained("prajjwal1/bert-tiny")


def test_trainset_bookcorpusnwiki(bookscorpusandwiki, snapshot):
    """first batch of trainset (BooksCorpusAndWiki) should match snapshot"""
    trainset = bookscorpusandwiki.datasets["train"]
    next_batch = next(iter(trainset))
    assert all(key in next_batch.keys() for key in ["input_ids", "attention_mask", "special_tokens_mask"])
    assert next_batch["input_ids"].shape == snapshot
    assert next_batch == snapshot


def test_trainloader_bookcorpusnwiki(bookscorpusandwiki, snapshot):
    """first batch of trainloader (BooksCorpusAndWiki) should match snapshot"""
    trainloader = bookscorpusandwiki.dataloader("train")
    next_batch = next(iter(trainloader))
    assert all(key in next_batch.keys() for key in ["input_ids", "attention_mask", "labels"])
    assert next_batch["input_ids"].shape == snapshot
    assert next_batch == snapshot


def test_trainingargs_bookcorpusnwiki(bookscorpusandwiki, model, snapshot):
    """training arguments for BooksCorpusAndWiki should match snapshot"""
    required_args = {"output_dir": "/tmp", "max_steps": int(1e5), "logging_dir": "/tmp/runs"}
    args = TrainingArguments(**bookscorpusandwiki.training_args, **required_args)
    assert args == snapshot

    # Don't snapshot-test Trainer. It has hardware-specific configs, so it's not portable.
    assert Trainer(model, args, **bookscorpusandwiki.trainer_params)

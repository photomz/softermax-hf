import pytest
from transformers import RobertaTokenizer

from src.dataloader import BooksCorpusAndWiki


# These tests require Internet to download the dataset, or a precached dataset.
@pytest.fixture(scope="module")
def bookscorpusandwiki():
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    loader = BooksCorpusAndWiki(tokenizer)
    loader.setup()
    return loader


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

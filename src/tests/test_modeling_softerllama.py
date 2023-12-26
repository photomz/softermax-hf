"""
PyTest unit testing. Run the unit tests from the root directory using
`python -m pytest tests/test_modeling_softerllama.py`
"""

import pytest
from pytest import fixture
from src import SofterLlamaConfig, SofterLlamaForCausalLM
from transformers import LlamaConfig, LlamaForCausalLM, LlamaTokenizerFast
import torch


@fixture(scope="session")
def tokenizer() -> LlamaTokenizerFast:
    return LlamaTokenizerFast.from_pretrained("nickypro/tinyllama-15M")


@fixture(scope="session")
def softerllama_config() -> SofterLlamaConfig:
    # adopts 15M model's params
    sl_config = SofterLlamaConfig.from_pretrained("nickypro/tinyllama-15M")
    sl_config.n_bias = 1
    return sl_config


@fixture(scope="session")
def sl_model(softerllama_config) -> SofterLlamaForCausalLM:
    return SofterLlamaForCausalLM.from_pretrained("models/softermax-tinyllama-15m.pt", config=softerllama_config).eval()


def test_softermax0_equal_softmax(tokenizer):
    og_model = LlamaForCausalLM.from_pretrained("nickypro/tinyllama-15M").eval()

    sl_0_config = SofterLlamaConfig.from_pretrained("nickypro/tinyllama-15M")
    sl_0_config.n_bias = 0
    sl_model = SofterLlamaForCausalLM.from_pretrained("nickypro/tinyllama-15M", config=sl_0_config).eval()
    assert sl_model.config.n_bias == 0

    inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
    og_logits = og_model(**inputs).logits.detach().cpu()
    sl_logits = sl_model(**inputs).logits.detach().cpu()
    # check that the all logits are equal, with absolute epsilon of 1e-4
    assert torch.allclose(og_logits, sl_logits, atol=1e-4)


def test_softermax(sl_model, tokenizer):
    prompt = "Sally went to the seashore"
    inputs = tokenizer(prompt, return_tensors="pt")

    # Generate
    generate_ids = sl_model.generate(inputs.input_ids, max_length=30)
    out = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

    # suspend input capture by py.test so user input can be recorded here
    capture_manager = pytest.config.pluginmanager.getplugin("capturemanager")
    capture_manager.suspendcapture(in_=True)

    answer = input(f"model generated: {out}\nPasses test? (y/n): ")

    # resume capture after question have been asked
    capture_manager.resumecapture()

    assert answer == "y"


def test_loading_and_saving(tmpdir, sl_model):
    sl_model.save_pretrained(tmpdir.join("/tinyllama-15M"), from_pt=True)

    loaded_model = SofterLlamaForCausalLM.from_pretrained(tmpdir.join("/tinyllama-15M"))
    assert loaded_model.config.n_bias == sl_model.config.n_bias


def test_trainer_evaluate():
    # waiting on implementation of dataloader
    # this function will pass a dataset validation batch through
    # the softertrainer evaluate function to verify
    # the evaluation loop is correct
    return

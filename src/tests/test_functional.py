import pytest
import torch
import torch.nn.functional as F
from torch import tensor

from src.functional import clip_logits, softermax

# important: so randn is determinstic every time. if not CPU, set seed on your device.
torch.manual_seed(42)


@pytest.mark.parametrize("n_bias", [0, 0.001, 1, 10])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("shape", [[50], [10, 10], [2, 2, 2, 2]])
def test_softermax(snapshot, n_bias, dtype, shape):
    """should match snapshot"""
    input = torch.randn(*shape, dtype=dtype)
    output = softermax(input, dim=-1, n_bias=tensor(n_bias))
    # same as previous snapshot, no change. run `pytest --snapshot-update` if function change is deliberate
    assert output == snapshot
    assert output.sum(dim=-1) == snapshot


def test_softermax_no_bias():
    """should only equal F.softmax at n_bias = 0"""
    input = torch.randn(10, 10)

    # test for n_bias = 0
    softer_output = softermax(input, dim=-1, n_bias=tensor(0))
    torch_output = F.softmax(input, dim=-1)

    assert torch.allclose(softer_output, torch_output)

    # should differ for n_bias = 0.01
    softer_output = softermax(input, dim=-1, n_bias=tensor(0.01))
    assert (softer_output <= torch_output).all()


def test_softermax_zeros():
    """input len = 9 at n_bias = 1 should sum to 9/10"""
    input = torch.zeros(9)

    output = softermax(input, n_bias=tensor(1))
    assert torch.isclose(output.sum(), tensor(0.9))


@pytest.mark.parametrize("gamma", [0, -0.001, -1, -10])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("shape", [[50], [10, 10], [2, 2, 2, 2]])
def test_clipped_softmax(snapshot, gamma, dtype, shape):
    """should match snapshot"""
    input = torch.randn(*shape, dtype=dtype)
    # eta = 1, so no upper clipping, Section 5.1 in http://arxiv.org/abs/2306.12929
    output = clip_logits(F.softmax(input, dim=-1), gamma=tensor(gamma), eta=tensor(1.0))

    assert output == snapshot
    assert output.sum(dim=-1) == snapshot


def test_softmax_no_clipping():
    """should only equal F.softmax at gamma = 0, eta = 1"""
    input = torch.randn(10, 10)

    # test for gamma = 0
    torch_output = F.softmax(input, dim=-1)
    clipped_output = clip_logits(torch_output, gamma=tensor(0), eta=tensor(1))

    assert (clipped_output == torch_output).all()

    # should differ for gamma = -0.01
    clipped_output = clip_logits(torch_output, gamma=tensor(-0.01), eta=tensor(1))
    # assert not torch.allclose(clipped_output, torch_output)
    assert (clipped_output <= torch_output).all()


def test_clipped_softmax_fp16():
    """F.softmax should throw RuntimeError at dtype = fp16, not supported."""
    input = torch.randn(10, 10, dtype=torch.float16)

    with pytest.raises(RuntimeError):
        clip_logits(F.softmax(input, dim=-1), gamma=tensor(0), eta=tensor(1))


def test_softmax_exact_clip():
    """should clip the way math says it should"""
    input = torch.tensor([[0.5] * 2 + [-1e10] * 8, [0.1] * 10]).float()

    # If 0.5 clips to 1 and 0.1 clips to 0, then, as per the paper (Section 4.1):
    # 0.5 = (1-y) / (l - y)
    # 0.1 = -y / (l - y)
    # where l = 9/4, y = -1/4.
    eta, gamma = 9 / 4, -1 / 4

    output = clip_logits(F.softmax(input, dim=-1), gamma=tensor(gamma), eta=tensor(eta))
    assert torch.allclose(output, torch.tensor([[1] * 2 + [0] * 8, [0] * 10]).float())

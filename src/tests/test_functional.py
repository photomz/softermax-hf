import pytest
import torch
import torch.nn.functional as F

from src.functional import softermax, clipped_softmax

torch.manual_seed(42)


@pytest.mark.parametrize("n_bias", [0, 0.001, 1, 10])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("shape", [[50], [10, 10], [2, 2, 2, 2]])
def test_softermax(snapshot, n_bias, dtype, shape):
    """should match snapshot"""
    tensor = torch.randn(*shape, dtype=dtype)
    output = softermax(tensor, dim=-1, n_bias=n_bias, dtype=dtype)
    # same as previous snapshot, no change. run `pytest --snapshot-update` if function change is deliberate
    assert output == snapshot
    assert output.sum(dim=-1) == snapshot


def test_softermax_no_bias():
    """should only equal F.softmax at n_bias = 0"""
    tensor = torch.randn(10, 10)

    # test for n_bias = 0
    manual_output = softermax(tensor, dim=-1, n_bias=0)
    torch_output = F.softmax(tensor, dim=-1)

    assert torch.allclose(manual_output, torch_output)

    # should differ for n_bias = 0.01
    manual_output = softermax(tensor, dim=-1, n_bias=0.01)
    assert not torch.allclose(manual_output, torch_output)


def test_softermax_zeros():
    """input len = 9 at n_bias = 1 should sum to 9/10"""
    tensor = torch.zeros(9)

    output = softermax(tensor, n_bias=1)
    assert torch.isclose(output.sum(), torch.tensor(0.9))


@pytest.mark.parametrize("gamma", [0, -0.001, -1, -10])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("shape", [[50], [10, 10], [2, 2, 2, 2]])
def test_clipped_softmax(snapshot, gamma, dtype, shape):
    """should match snapshot"""
    tensor = torch.randn(*shape, dtype=dtype)
    # eta = 1, so no upper clipping, Section 5.1 in http://arxiv.org/abs/2306.12929
    output = clipped_softmax(tensor, dim=-1, gamma=gamma, eta=1, dtype=dtype)

    assert output == snapshot
    assert output.sum(dim=-1) == snapshot


def test_softmax_no_clipping():
    """should only equal F.softmax at gamma = 0, eta = 1"""
    tensor = torch.randn(10, 10)

    # test for gamma = 0
    manual_output = clipped_softmax(tensor, dim=-1, gamma=0, eta=1)
    torch_output = F.softmax(tensor, dim=-1)

    assert torch.allclose(manual_output, torch_output)

    # should differ for gamma = -0.01
    manual_output = clipped_softmax(tensor, dim=-1, gamma=-0.01, eta=1)
    assert not torch.allclose(manual_output, torch_output)


def test_clipped_softmax_fp16():
    """F.softmax should throw RuntimeError at dtype = fp16, not supported."""
    tensor = torch.randn(10, 10, dtype=torch.float16)

    with pytest.raises(RuntimeError):
        clipped_softmax(tensor, dim=-1, gamma=0, eta=1)


def test_softmax_exact_clip():
    """should clip the way math says it should"""
    tensor = torch.tensor([[0.5] * 2 + [-1e10] * 8, [0.1] * 10]).float()

    # If 0.5 clips to 1 and 0.1 clips to 0, then, as per the paper (Section 4.1):
    # 0.5 = (1-y) / (l - y)
    # 0.1 = -y / (l - y)
    # where l = 9/4, y = -1/4.
    eta, gamma = 9 / 4, -1 / 4

    output = clipped_softmax(tensor, dim=-1, gamma=gamma, eta=eta)
    assert torch.allclose(output, torch.tensor([[1] * 2 + [0] * 8, [0] * 10]).float())

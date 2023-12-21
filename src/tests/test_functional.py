import pytest
import torch
import torch.nn.functional as F

from src.functional import softermax

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

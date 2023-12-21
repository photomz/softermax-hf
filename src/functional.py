"""
softermax implementation
"""

from typing import Optional
from torch.types import _dtype
import torch
from torch import Tensor


# Ref: https://github.com/softmax1/EsperBERTo/blob/946f0a9fa0f6e3b2bf755388d8fa20c31f8e2bf5/src/functional.py#L49
def softermax(
    input: Tensor,
    n_bias: float = 0.0,
    dim: Optional[int] = -1,
    dtype: Optional[_dtype] = None,
) -> Tensor:
    """
    $\text(softmax)_n(x_i) = exp(x_i) / (n + \sum_j exp(x_j))$

    Note: softmax_n, with fixed input, is _not_ shift-symmetric when n != 0, and we must account for this.
    Normally when computing a softmax, the maxes are subtracted from the inputs for numeric stability.
    """
    # compute the maxes along the last dimension
    input_maxes = input.max(dim=dim, keepdim=True).values.detach()
    # shift the input to prevent overflow (and underflow in the denominator)
    # compute the numerator and softmax_0 denominator using the shifted input
    numerator = torch.exp(input - input_maxes)
    denominator = numerator.sum(dim=dim, keepdim=True)
    # we need to shift the zeros in the same way we shifted the inputs
    shifted_bias = n_bias * torch.exp(-input_maxes)
    # and then add this contribution to the denominator
    scores = numerator / (denominator + shifted_bias)
    return scores if dtype is None else scores.type(dtype=dtype)

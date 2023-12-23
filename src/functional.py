"""
softermax implementation
"""

from typing import Optional

import torch
import torch.nn.functional as F

zero = torch.tensor(0.0)
one = torch.tensor(1.0)


# Ref: https://github.com/softmax1/EsperBERTo/blob/946f0a9fa0f6e3b2bf755388d8fa20c31f8e2bf5/src/functional.py#L49
def softermax(input: torch.Tensor, n_bias: torch.Tensor = zero, dim: Optional[int] = -1) -> torch.Tensor:
    """
    $\text(softmax)_n(x_i) = exp(x_i) / (n + \sum_j exp(x_j))$

    Note: softmax_n, with fixed input, is _not_ shift-symmetric when n != 0, and we must account for this.
    Normally when computing a softmax, the maxes are subtracted from the inputs for numeric stability.
    """
    # a learnable bias must be in Tensor format for autograd
    assert n_bias.numel() == 1, "must be single-element Tensor"
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
    return scores


# Ref: https://github.com/Qualcomm-AI-research/outlier-free-transformers/blob/1e59744972da808a90d9996027bfc274689594b8/transformers_language/models/softmax.py#L8
def clip_logits(score: torch.Tensor, gamma: torch.Tensor = zero, eta: torch.Tensor = one) -> torch.Tensor:
    """clip((eta - gamma) * input + gamma) in [0,1], eta >= 1, gamma <= 0.

    Exact zeros can fix outliers, so fix eta = 1 (no upper clipping) and vary gamma (lower clipping).
    Normal softmax is clip_logits(F.softmax(x), eta=1, gamma=0).
    """
    assert gamma.numel() == 1 and eta.numel() == 1, "must be single-element Tensor"
    # Stretch and translate by linear factor
    stretched_score = score * (eta - gamma) + gamma
    # Clip each score to [0,1], so sum != 1 anymore.
    # In fact, every score in attention sum can be 0 or 1.
    clipped_score = torch.clip(stretched_score, 0, 1)
    return clipped_score


def clipped_softmax(input: torch.Tensor, dim: int = -1, gamma: float = 0.0, eta: float = 1.0, **kwargs) -> torch.Tensor:
    # F.softmax doesn't support fp16 on CPU
    score = F.softmax(input, dim=dim, **kwargs)
    stretched_score = score * (eta - gamma) + gamma
    clipped_score = torch.clip(stretched_score, 0, 1)
    return clipped_score

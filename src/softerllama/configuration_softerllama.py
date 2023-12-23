"""
Subclasses the original llama config to add softermax denominator n_bias
"""

from transformers.models.llama import LlamaConfig


class SofterLlamaConfig(LlamaConfig):
    def __init__(self, n_bias: float = 0.0, n_clip: float = 0.0, learn_softmax: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.n_bias = n_bias  # Softermax bias
        self.n_clip = n_clip  # Clipped softmax gamma
        self.learn_softmax = learn_softmax  # Is softmax's param (bias/clip) learnable?

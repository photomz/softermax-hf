"""
Subclasses the original llama config to add softermax denominator n_bias
"""

from transformers.models.llama import LlamaConfig


class SofterLlamaConfig(LlamaConfig):
    def __init__(self, n_bias: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self.n_bias = n_bias

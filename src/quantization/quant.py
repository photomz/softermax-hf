"""
Standalone quantization code. Contains a globally referencable dictionary of
default quantization config settings and a function that will quantize a given model.
"""
from transformers import PreTrainedModel, GPTQConfig
from .quantizer import GPTQQuantizer
from torch.utils.data import DataLoader
import torch

# default quantization configuration, requires the dataset be specified
default_quant_configs = {
    "bits": 4,
    "group_size": 128,
    "desc_act": True,
    "disable_exllama": True,
}


def quantize(model: PreTrainedModel, quant_config: dict):
    quantizer = GPTQQuantizer.from_dict(quant_config)
    quantized_model = quantizer.quantize_model(model.to(torch.float16))
    return quantized_model

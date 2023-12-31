"""
Standalone quantization code. Contains a globally referencable dictionary of
default quantization config settings and a function that will quantize a given model.
"""
from transformers import PreTrainedModel, GPTQConfig
from torch.utils.data import DataLoader

# default quantization configuration, requires the dataset be specified
default_quant_configs = {"bits": 4, "group_size": 128, "desc_act": False}


def quantize(model: PreTrainedModel, quant_config: GPTQConfig):
    # quantization entry point in huggingface is done through the from_pretrained
    # instantiation of a new model. Instead of saving the whole model to disk we just
    # pass in the state dict of the existing model in memory to initialise a
    # quantized version of it.
    modelclass = model.__class__

    quant_model = modelclass.from_pretrained(
        state_dict=model.state_dict(), quantization_config=quant_config, device_map="auto"
    )
    return quant_model

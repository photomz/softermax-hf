"""
Should contain quantization code for @MZ to implement. For now using placeholder
functions for SofterTrainer implementation.
"""


def quantize(model):
    # note: MZ pls ensure model is returned on the device on which
    # inference is to be run (GPU if supported, else CPU)
    return model

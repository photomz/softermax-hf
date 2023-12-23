"""
Holds utility functions and custom classes for training
of the softermax models.
"""

from typing import Optional, List, Dict
from torch.utils.data import Dataset

from transformers import Trainer
from transformers.utils import logging

from .quantization.quant import quantize


logger = logging.get_logger(__name__)


class SofterTrainer(Trainer):
    """
    Subclass of the original Trainer to include custom evaluation code for quantized and unquantized models.
    """

    def evaluate(
        self,
        eval_dataset: Optional[Dataset] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> Dict[str, float]:
        """
        Override default evaluate loop to run evaluation metrics for both quantized and unquantized versions of the model.


        Returns:
            A dictionary containing the evaluation loss and the potential metrics computed from the predictions. The
            dictionary also contains the epoch number which comes from the training state.
        """
        # first run regular model eval
        fp16_metrics = super().evaluate(eval_dataset, ignore_keys, metric_key_prefix="fp16")

        # store a ref to the original model while moving it to cpu
        fp16_model = self.model.to("cpu")
        # quantize the model
        logger.info("Quantizing model for eval loop")
        int8_model = quantize(self.model)
        # replace current ref to the quantized model
        self.model = int8_model

        # then run quant model eval
        int8_metrics = super().evaluate(eval_dataset, ignore_keys, metric_key_prefix="int8")

        # return self.model ref to regular model
        self.model = fp16_model

        # combine both metric dictionaries into one
        metrics_dict = fp16_metrics.update(int8_metrics)
        return metrics_dict


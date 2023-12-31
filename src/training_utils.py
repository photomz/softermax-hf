"""
Holds utility functions and custom classes for training
of the softermax models.
"""

from typing import Optional, List, Dict
from torch.utils.data import Dataset

from dataclasses import dataclass, field
from transformers import Trainer, TrainingArguments, GPTQConfig
from transformers.utils import logging

from .quantization.quant import quantize, default_quant_configs


logger = logging.get_logger(__name__)


@dataclass
class SofterTrainingArguments(TrainingArguments):
    """
    Subclass of TrainingArguments to add a field for the quantization configuration
    """

    quant_kwargs: dict = field(
        default_factory=default_quant_configs,
        metadata={"help": "GPTQConfig-specific **kwargs"},
    )


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
        # store original output_attentions boolean value
        orig_output_attentions = self.model.config.output_attentions
        # ensure model will output attention matrix for compute_metrics
        self.model.config.output_attentions = True

        # first run regular model eval
        print("regular model eval")
        orig_metrics = super().evaluate(eval_dataset, ignore_keys, metric_key_prefix="orig")

        # store a ref to the original model while moving it to cpu
        orig_model = self.model.to("cpu")
        # quantize the model
        print("quantizing")
        logger.info("Quantizing model for eval loop")
        
        quant_config = GPTQConfig(dataset=self.eval_dataset, **self.args.quant_kwargs)
        quant_model = quantize(self.model, quant_config)
        # replace current ref to the quantized model
        self.model = quant_model

        # then run quant model eval
        print("quant model eval")
        quant_metrics = super().evaluate(eval_dataset, ignore_keys, metric_key_prefix="quant")

        # return self.model ref to regular model
        self.model = orig_model
        # reset output_attentions to its original value
        self.model.config.output_attentions = orig_output_attentions

        # combine both metric dictionaries into one
        metrics_dict = orig_metrics.update(quant_metrics)
        return metrics_dict


def compute_softermetrics(eval_preds):
    print("compute metrics")
    print(eval_preds)

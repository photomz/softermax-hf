"""
Holds utility functions and custom classes for training
of the softermax models.
"""

from typing import Optional, Union, List, Dict
from torch.utils.data import Dataset
import torch

from dataclasses import dataclass, field
from transformers import Trainer, TrainingArguments, PreTrainedTokenizerBase, GPTQConfig
from transformers.utils import logging

# from .quantization.quant import quantize, default_quant_configs
# from .quantization.quantizer import GPTQQuantizer


logger = logging.get_logger(__name__)


@dataclass
class SofterTrainingArguments(TrainingArguments):
    """
    Subclass of TrainingArguments to add a field for the quantization configuration
    """

    """
    quant_kwargs: dict = field(
        default_factory=default_quant_configs,
        metadata={"help": "GPTQConfig-specific **kwargs"},
    )
    quant_dataset: Union[list, str] = field(
        default="c4",
        metadata={
            "help": "GPTQConfig only accepts calibration datasets as list of strings, otherwise use the default c4 dataset"
        },
    )
    quant_tokenizer: PreTrainedTokenizerBase = field(
        default=None,
        metadata={
            "help": "Tokenizer for tokenizing calibration dataset, should be the same as tokenizer used in the dataloader"
        },
    )
    """


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

        """
        Quant code not working. Prioritise getting train loop up and running first.
        # store a ref to the original model while moving it to cpu using inbuilt Trainer function
        # TODO: double check this works on DDP system
        orig_model = self.model
        self._move_model_to_device(orig_model, "cpu")
        print(next(orig_model.parameters()).device)
        # quantize the model
        print("quantizing")
        logger.info("Quantizing model for eval loop")
        quantizer = GPTQQuantizer(dataset="c4", **self.args.quant_kwargs)
        quantized_model = self.model.to(torch.float16)
        self._move_model_to_device(quantized_model, "cuda")
        quantized_model = quantizer.quantize_model(quantized_model, self.args.quant_tokenizer)
        # replace current ref to the quantized model
        self.model = quantized_model

        # then run quant model eval
        print("quant model eval")
        quant_metrics = super().evaluate(eval_dataset, ignore_keys, metric_key_prefix="quant")

        # return self.model ref to regular model and move to gpu
        self.model = orig_model
        self._move_model_to_device(orig_model, self.args.device)
        """
        # reset output_attentions to its original value
        self.model.config.output_attentions = orig_output_attentions

        # combine both metric dictionaries into one
        # orig_metrics.update(quant_metrics)

        return orig_metrics


def compute_softermetrics(eval_preds):
    print("compute metrics")
    print(eval_preds)


def preprocess_logits_for_metrics(logits, labels):
    """
    Original Trainer may have a memory leak.
    This is a workaround to avoid storing too many tensors that are not needed.
    """
    print(logits)
    print(labels)
    pred_ids = torch.argmax(logits[0], dim=-1)
    return pred_ids, labels

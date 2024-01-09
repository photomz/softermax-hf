"""
Holds utility functions and custom classes for training
of the softermax models.
"""

from typing import Optional, Union, List, Dict
from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np

from dataclasses import dataclass, field

from transformers import Trainer, TrainingArguments, PreTrainedTokenizerBase, GPTQConfig
from transformers.utils import logging
from transformers.integrations.deepspeed import deepspeed_init, deepspeed_load_checkpoint, is_deepspeed_available
from transformers.trainer_utils import EvalLoopOutput, EvalPrediction, has_length, denumpify_detensorize
from transformers.trainer_pt_utils import IterableDatasetShard, find_batch_size, nested_concat, nested_numpify
from transformers.utils import is_torch_tpu_available

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
        print(self.model.config.output_attentions)

        # first run regular model eval
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

    # TODO subclass trainer evaluation_loop. currently their behaviour is to collate all relevant metric tensors into one
    # really large tensor on on GPU (or move to CPU with accumulation steps), this causes memory to blow up and kill the process
    # we have to change the eval loop to make compute_metrics be called on the fly, at the time that the tensors would have been
    # moved to the CPU, we just run the metrics then discard the tensors since our attention sum metrics don't require the whole
    # dataset to be evaluated first.
    def evaluation_loop(
        self,
        dataloader: DataLoader,
        description: str,
        prediction_loss_only: Optional[bool] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> EvalLoopOutput:
        """
        Prediction/evaluation loop, shared by `Trainer.evaluate()` and `Trainer.predict()`.

        Subclassed and changed compute_metrics behaviour. Original Trainer would collate all relevant tensors
        onto CPU and then run the compute_metrics on very large tensors. This is infeasible for us because
        1. we require entire attention matrices for our metrics
        2. compute_metrics won't even get called in vanilla Trainer if there are no labels supplied (which don't exist in decoder-only models)
        3. we're broke
        In this new subclass, compute_metrics is called every 'eval_accumulation_steps' to prevent the large validation dataset that we
        have from killing the process due to exceeding memory capacity, and this will work so long as every tensor can have their metrics
        computed independently from each other.

        Note: There are quite a few changes to unnecessary functionality such as TPU support and
        """
        args = self.args

        prediction_loss_only = prediction_loss_only if prediction_loss_only is not None else args.prediction_loss_only

        # if eval is called w/o train, handle model prep here
        if self.is_deepspeed_enabled and self.deepspeed is None:
            _, _ = deepspeed_init(self, num_training_steps=0, inference=True)

        model = self._wrap_model(self.model, training=False, dataloader=dataloader)

        if len(self.accelerator._models) == 0 and model is self.model:
            model = (
                self.accelerator.prepare(model)
                if self.is_deepspeed_enabled
                else self.accelerator.prepare_model(model, evaluation_mode=True)
            )

            if self.is_fsdp_enabled:
                self.model = model

            # for the rest of this function `model` is the outside model, whether it was wrapped or not
            if model is not self.model:
                self.model_wrapped = model

            # backward compatibility
            if self.is_deepspeed_enabled:
                self.deepspeed = self.model_wrapped

        # if full fp16 or bf16 eval is wanted and this ``evaluation`` or ``predict`` isn't called
        # while ``train`` is running, cast it to the right dtype first and then put on device
        if not self.is_in_train:
            if args.fp16_full_eval:
                model = model.to(dtype=torch.float16, device=args.device)
            elif args.bf16_full_eval:
                model = model.to(dtype=torch.bfloat16, device=args.device)

        batch_size = self.args.eval_batch_size

        logger.info(f"***** Running {description} *****")
        if has_length(dataloader):
            logger.info(f"  Num examples = {self.num_examples(dataloader)}")
        else:
            logger.info("  Num examples: Unknown")
        logger.info(f"  Batch size = {batch_size}")

        model.eval()

        self.callback_handler.eval_dataloader = dataloader
        if args.past_index >= 0:
            self._past = None

        # Initialize containers
        # losses/preds/labels on GPU/TPU (accumulated for eval_accumulation_steps)
        losses_host = None
        preds_host = None
        labels_host = None
        inputs_host = None

        # Will be useful when we have an iterable dataset so don't know its length.
        observed_num_examples = 0

        # Main evaluation loop
        all_metrics = {}
        for step, inputs in enumerate(dataloader):
            # Update the observed num examples
            observed_batch_size = find_batch_size(inputs)
            if observed_batch_size is not None:
                observed_num_examples += observed_batch_size
                # For batch samplers, batch_size is not known by the dataloader in advance.
                if batch_size is None:
                    batch_size = observed_batch_size

            # Prediction step
            loss, logits, labels = self.prediction_step(model, inputs, prediction_loss_only, ignore_keys=ignore_keys)
            main_input_name = getattr(self.model, "main_input_name", "input_ids")
            inputs_decode = self._prepare_input(inputs[main_input_name]) if args.include_inputs_for_metrics else None

            # Update containers on host
            if loss is not None:
                losses = self.gather_function((loss.repeat(batch_size)))
                losses_host = losses if losses_host is None else nested_concat(losses_host, losses, padding_index=-100)
            if labels is not None:
                labels = self.accelerator.pad_across_processes(labels, dim=1, pad_index=-100)
            if inputs_decode is not None:
                inputs_decode = self.accelerator.pad_across_processes(inputs_decode, dim=1, pad_index=-100)
                inputs_decode = self.gather_function((inputs_decode))
                inputs_host = (
                    inputs_decode
                    if inputs_host is None
                    else nested_concat(inputs_host, inputs_decode, padding_index=-100)
                )
            if logits is not None:
                logits = self.accelerator.pad_across_processes(logits, dim=1, pad_index=-100)
                if self.preprocess_logits_for_metrics is not None:
                    logits = self.preprocess_logits_for_metrics(logits, labels)
                logits = self.gather_function((logits))
                preds_host = logits if preds_host is None else nested_concat(preds_host, logits, padding_index=-100)
            if labels is not None:
                labels = self.gather_function((labels))
                labels_host = labels if labels_host is None else nested_concat(labels_host, labels, padding_index=-100)

            self.control = self.callback_handler.on_prediction_step(args, self.state, self.control)

            # Gather all tensors and run metric computation if we have done enough accumulation steps, then discard the saved tensors
            if args.eval_accumulation_steps is not None and (step + 1) % args.eval_accumulation_steps == 0:
                if losses_host is not None:
                    losses = nested_numpify(losses_host)
                if preds_host is not None:
                    logits = nested_numpify(preds_host)
                if inputs_host is not None:
                    inputs_decode = nested_numpify(inputs_host)
                if labels_host is not None:
                    labels = nested_numpify(labels_host)

                if args.include_inputs_for_metrics:
                    metrics = self.compute_metrics(
                        EvalPrediction(predictions=logits, label_ids=labels, inputs=inputs_decode)
                    )
                else:
                    metrics = self.compute_metrics(EvalPrediction(predictions=logits, label_ids=labels))

                all_metrics = (
                    metrics if all_metrics is None else nested_concat(all_metrics, metrics, padding_index=-100)
                )

                # Set back to None to begin a new accumulation
                losses_host, preds_host, inputs_host, labels_host = None, None, None, None

        # After all calls to `.gather_function`, reset to `gather_for_metrics`:
        self.gather_function = self.accelerator.gather_for_metrics
        if args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of the evaluation loop
            delattr(self, "_past")

        # Gather all remaining tensors and execute final compute_metrics
        if losses_host is not None:
            losses = nested_numpify(losses_host)
        if preds_host is not None:
            logits = nested_numpify(preds_host)
        if inputs_host is not None:
            inputs_decode = nested_numpify(inputs_host)
        if labels_host is not None:
            labels = nested_numpify(labels_host)
        if args.include_inputs_for_metrics:
            metrics = self.compute_metrics(EvalPrediction(predictions=logits, label_ids=labels, inputs=inputs_decode))
        else:
            metrics = self.compute_metrics(EvalPrediction(predictions=logits, label_ids=labels))
        all_metrics = metrics if all_metrics is None else nested_concat(all_metrics, metrics, padding_index=-100)

        # To be JSON-serializable, we need to remove numpy types or zero-d tensors
        all_metrics = denumpify_detensorize(all_metrics)

        if hasattr(self, "jit_compilation_time"):
            all_metrics[f"{metric_key_prefix}_jit_compilation_time"] = self.jit_compilation_time

        # Prefix all keys with metric_key_prefix + '_'
        for key in list(all_metrics.keys()):
            if not key.startswith(f"{metric_key_prefix}_"):
                all_metrics[f"{metric_key_prefix}_{key}"] = all_metrics.pop(key)

        # we only care about metrics being returned, neither predictions nor label_ids are used in evaluate() anyway
        return EvalLoopOutput(predictions=None, label_ids=None, metrics=metrics, num_samples=observed_num_examples)


def compute_softermetrics(eval_preds):
    """
    eval_preds is expected to be EvalPrediction object.
    Ref: https://huggingface.co/docs/transformers/main_classes/output#transformers.modeling_outputs.CausalLMOutput
    [0] is the output raw logits of shape (batch_size, seq_len, vocab_size)
    [1] are the attention matrices. Tuple of size num_layers, each element in the tuple being (batch_size, num_heads, seq_len, seq_len)
    """
    # TODO, figure out what metrics we want
    return {}

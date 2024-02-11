"""
Holds utility functions and custom classes for training
of the softermax models.
"""

from typing import Optional, Union, List, Dict
from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
import pandas as pd

from dataclasses import dataclass, field

from transformers import Trainer, TrainingArguments, TrainerState, TrainerControl, PreTrainedTokenizerBase, GPTQConfig
from transformers.utils import logging
from transformers.integrations.deepspeed import deepspeed_init, deepspeed_load_checkpoint, is_deepspeed_available
from transformers.integrations import WandbCallback
from transformers.trainer_utils import EvalLoopOutput, EvalPrediction, has_length, denumpify_detensorize
from transformers.trainer_pt_utils import IterableDatasetShard, find_batch_size, nested_concat, nested_numpify
from transformers.utils import is_torch_tpu_available

import wandb

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
        # first run regular model eval
        orig_metrics = super().evaluate(eval_dataset, ignore_keys, metric_key_prefix="eval_orig")

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

        # combine both metric dictionaries into one
        orig_metrics.update(quant_metrics)
        """

        return orig_metrics

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
        1. compute_metrics won't even get called in vanilla Trainer if there are no labels supplied
        2. we don't have the GPU memory to hold everything in
        3. we're broke
        In this new subclass, compute_metrics is called every 'eval_accumulation_steps' to prevent the large validation dataset that we
        have from killing the process due to exceeding memory capacity, and this will work so long as every tensor can have their metrics
        computed independently from each other. compute_metrics is also expected to be a wrapper function like wandb_metric_computer that I
        created below.

        Note: There are quite a few changes to unnecessary functionality such as TPU support and nothing is guaranteed to work
        (honestly I'm surprised anything I write runs)
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
        all_metrics = None
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
                # for metric computation, while huggingface uses EvalPrediction, i'm going to explicitly pass a dictionary
                # instead, because default behaviour is to not provide the loss as input to compute_metrics, but we need
                # the loss in order to calculate the perplexity
                if args.include_inputs_for_metrics:
                    metrics = self.compute_metrics(
                        {"loss": losses, "predictions": logits, "label_ids": labels, "inputs": inputs_decode}
                    )
                else:
                    metrics = self.compute_metrics({"loss": losses, "predictions": logits, "label_ids": labels})
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
        # for metric computation, while huggingface uses EvalPrediction, i'm going to explicitly pass a dictionary
        # instead, because default behaviour is to not provide the loss as input to compute_metrics, but we need
        # the loss in order to calculate the perplexity
        if args.include_inputs_for_metrics:
            metrics = self.compute_metrics(
                {"loss": losses, "predictions": logits, "label_ids": labels, "inputs": inputs_decode},
            )
        else:
            metrics = self.compute_metrics(
                {"loss": losses, "predictions": logits, "label_ids": labels},
            )
        all_metrics = metrics if all_metrics is None else nested_concat(all_metrics, metrics, padding_index=-100)
        # To be JSON-serializable, we need to remove numpy types or zero-d tensors
        all_metrics = denumpify_detensorize(all_metrics)

        if hasattr(self, "jit_compilation_time"):
            all_metrics[f"{metric_key_prefix}_jit_compilation_time"] = self.jit_compilation_time

        # Prefix all keys with metric_key_prefix + '_'
        for key in list(all_metrics.keys()):
            if not key.startswith(f"{metric_key_prefix}_"):
                all_metrics[f"{metric_key_prefix}_{key}"] = all_metrics.pop(key)

        # Usually EvalLoopOutput will return predictions and label_ids as the large concatenated tensor on CPU of all
        # the model outputs. Of course, here we don't create this large tensor, so there is nothing to output.
        # We only care about metrics being returned, neither predictions nor label_ids are used in .evaluate() anyway.
        # The only time the large tensors are returned is in Trainer.predict() which we won't use (but if we do, note this behaviour).
        return EvalLoopOutput(predictions=None, label_ids=None, metrics=all_metrics, num_samples=observed_num_examples)


def compute_softermetrics(eval_preds: dict):
    """
    eval_preds is expected to be dictionary object instead of the default EvalPredictions object because we overrode
    the original behaviour to pass in 'loss' as a metric. keys that are present change based on the config.
    ['loss'] is the loss.
    ['predictions'] is the output raw logits of shape (batch_size, seq_len, vocab_size)
    ['label_ids'] are inputs but right shifted
    ['inputs'] (if provided) are label_ids but not right shifted :)
    """
    logits = eval_preds["predictions"]
    label_ids = eval_preds["label_ids"]

    ppl = np.exp(eval_preds["loss"]).tolist()
    return {"ppl": ppl}


class WandbComputeMetricCallback(WandbCallback):
    """
    Have to manually log wandb tables using a wandbcallback, trying to do it through the compute_metrics function (original approach)
    throws an error because trainer fails to serialize wandb tables that are stored in state.log_history after being output by compute_metrics.

    The dataloader will extract the first batch from the validation dataset. This same first batch will always be used to evaluate softmax metrics
    every evaluation loop. Because the evaluation compute_metrics is completely disjoint from the wandb callback, we're going to have to use a
    couple dirty hacks and minor inefficiencies to get this to work properly.
    """

    def __init__(self, trainer: Trainer, tokenizer: PreTrainedTokenizerBase, val_dataloader):
        """Initializes the WandbPredictionProgressCallback instance.

        Args:
            trainer: The Hugging Face Trainer instance.
            tokenizer: The tokenizer associated with the model.
            val_dataloader (Dataloader): The validation dataloader.
        """
        super().__init__()
        self.trainer = trainer
        self.tokenizer = tokenizer
        self.sample_dataset = next(iter(val_dataloader))
        # special_tokens_mask causes prediction_step to throw an error because the model forward pass doesn't recognise it
        self.sample_dataset.pop("special_tokens_mask", None)

    def decode_predictions(self, logits, labels):
        labels = self.tokenizer.batch_decode(labels)
        logits = logits.argmax(axis=-1)
        prediction_text = self.tokenizer.batch_decode(logits)
        return {"labels": labels, "predictions": prediction_text}

    def on_evaluate(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        super().on_evaluate(args, state, control, **kwargs)

        # store original output_attentions boolean value
        orig_output_attentions = self.trainer.model.config.output_attentions
        # ensure model will output attention matrix for compute_metrics
        self.trainer.model.config.output_attentions = True

        # generate predictions
        # predictions[0] is the output raw logits of shape (batch_size, seq_len, vocab_size)
        # predictions[1] are the attention matrices. Tuple of size num_layers, each element in the tuple being (batch_size, num_heads, seq_len, seq_len)
        loss, predictions, labels = self.trainer.prediction_step(
            self.trainer.model, self.sample_dataset, prediction_loss_only=False, ignore_keys=None
        )
        logits, attn_matrices = predictions

        # reset output_attentions to its original value
        self.trainer.model.config.output_attentions = orig_output_attentions

        # decode predictions and labels
        decoded_preds = self.decode_predictions(logits, labels)
        # add predictions to a wandb.Table
        columns = ["steps", "label", "prediction", "layer", "head", "softmax_sum"]
        predictions_table = wandb.Table(columns=columns)

        for layer_num, layer_attn in enumerate(attn_matrices):
            # should be of shape (batch_size, num_heads, seq_len, seq_len)
            for batch_num, batch_instance in enumerate(layer_attn):
                # iterates through the batch_size dim, now batch_instance should be (num_heads, seq_len, seq_len)
                # iterates through the num_heads dim, head should be (seq_len, seq_len) and the row should softmax sum to between 0-1
                for head_num, head in enumerate(batch_instance):
                    # note: wandb images follow PIL's cartesian pixel coordinate system, with (0,0) in the upper left corner
                    softmax_sum = wandb.Image(head)
                    predictions_table.add_data(
                        state.global_step,
                        decoded_preds["labels"][batch_num],
                        decoded_preds["predictions"][batch_num],
                        layer_num,
                        head_num,
                        softmax_sum,
                    )
        # log the table to wandb
        self._wandb.log({"sample_predictions": predictions_table})

        # metrics computed by the last evaluation phase
        # TODO: see what ppl looks like. figure out a way to log the avg, max, min, var of the final ppl list in compute_metrics
        metrics = kwargs["metrics"]
        print(f"metrics\n{metrics}")

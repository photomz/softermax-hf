"""
Train script for any models. Set the configs using a wandb configuration .yaml file
by running `python train.py --c configs/{config_file_here}.yaml`
"""

import os
import argparse
import wandb
from datetime import date
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim import AdamW

from src import SofterLlamaConfig, SofterLlamaForCausalLM
from src import SofterBertConfig, SofterBertForMaskedLM
from src.scheduler import WarmupDecayedCosineAnnealingWarmRestarts, get_param_groups
from src.dataloader import BooksCorpusAndWiki
from src.training_utils import SofterTrainer, SofterTrainingArguments, WandbComputeMetricCallback, compute_softermetrics
from transformers import LlamaTokenizerFast, BertTokenizer

# sets env variables for wandb, see: https://docs.wandb.ai/guides/integrations/huggingface#additional-wb-settings
os.environ["WANDB_LOG_MODEL"] = "checkpoint"
os.environ["WANDB_WATCH"] = "all"

# maps .yaml config file model names to model and tokenizer classes
model_mapping = {"softerllama": SofterLlamaForCausalLM, "softerbert": SofterBertForMaskedLM}
config_mapping = {"softerllama": SofterLlamaConfig, "softerbert": SofterBertConfig}
tokenizer_mapping = {"softerllama": LlamaTokenizerFast, "softerbert": BertTokenizer}

# config files are no longer automatically loaded with --configs arg, see: https://github.com/wandb/wandb/issues/5648#issuecomment-1764436879
# fine, i'll do it myself
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument(
    "-c",
    "--config",
    type=str,
    default="config-defaults.yaml",
    help="Path to a .yaml file defining the configuration for the run",
)
args = parser.parse_args()

wandb.init(project="training-run", entity="softermax", config=args.config)
wandb.run.name = f"{wandb.config.run_name}-{date.today()}"

# model configs setup
config = config_mapping[wandb.config.model_name].from_pretrained(wandb.config.model_config_src)
config.n_bias = wandb.config.n_bias

# tokenizer setup
tokenizer = tokenizer_mapping[wandb.config.model_name].from_pretrained(wandb.config.model_config_src)

# model setup
model = model_mapping[wandb.config.model_name](config=config)

# dataset setup
bookscorpusandwiki = BooksCorpusAndWiki(
    tokenizer,
    mlm_probability=wandb.config.mlm_probability,
    batch_size={"train": wandb.config.batch_size, "validation": wandb.config.eval_batch_size},
)
bookscorpusandwiki.setup()

# decayed cosine annealing with hard restarts scheduler setup
optimizer = AdamW(
    get_param_groups(model.named_parameters()),
    lr=float(wandb.config.learning_rate),
    betas=(wandb.config.adam_beta1, wandb.config.adam_beta2),
    eps=wandb.config.adam_epsilon,
    weight_decay=wandb.config.weight_decay,
)
scheduler = WarmupDecayedCosineAnnealingWarmRestarts(
    optimizer,
    warmup_iters=int(wandb.config.warmup_steps),
    T_0=int(wandb.config.num_iter_per_restart),
    decay=float(wandb.config.peak_decay),
)

# trainer setup
training_args = SofterTrainingArguments(
    output_dir=wandb.config.output_dir,
    max_steps=wandb.config.total_steps,
    evaluation_strategy="steps",
    eval_steps=wandb.config.eval_steps,
    eval_accumulation_steps=wandb.config.eval_accumulation_steps,
    save_strategy="steps",
    save_steps=wandb.config.save_steps,
    per_device_train_batch_size=wandb.config.batch_size,
    per_device_eval_batch_size=wandb.config.eval_batch_size,
    gradient_accumulation_steps=wandb.config.grad_accum_steps,
    logging_steps=wandb.config.logging_steps,
    report_to="wandb",
    seed=wandb.config.seed,
    **bookscorpusandwiki.training_args,
)

trainer = SofterTrainer(
    model,
    training_args,
    optimizers=(optimizer, scheduler),
    compute_metrics=compute_softermetrics,
    **bookscorpusandwiki.trainer_params,
)

# Instantiate the WandbComputeMetricProgressCallback
wandb_compute_metrics_callback = WandbComputeMetricCallback(
    config_filepath=args.config
    trainer=trainer,
    tokenizer=tokenizer,
    val_dataloader=bookscorpusandwiki.dataloader("validation"),
)
# Add the callback to the trainer
trainer.add_callback(wandb_compute_metrics_callback)
# run the training
trainer.train()

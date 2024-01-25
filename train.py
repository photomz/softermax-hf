"""
Train script for any models. Set the configs using a wandb configuration .yaml file
by running `python train.py --configs configs/{config_file_here}.yaml`
"""

import argparse
import wandb
from torch.optim.lr_scheduler import OneCycleLR
from torch.optim import AdamW

from src import SofterLlamaConfig, SofterLlamaForCausalLM
from src import SofterBertConfig, SofterBertForMaskedLM
from src.dataloader import BooksCorpusAndWiki
from src.training_utils import SofterTrainer, SofterTrainingArguments, wandb_metric_computer

from transformers import LlamaTokenizerFast, BertTokenizer

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

wandb.init(project="training-runs", entity="softermax", config=args.config)
wandb.run.name = wandb.config.run_name

# model configs setup
config = config_mapping[wandb.config.model_name].from_pretrained(wandb.config.model_config_src)
config.n_bias = 1

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

# onecycle learning rate scheduler setup
optimizer = AdamW(
    model.parameters(),
    betas=(wandb.config.adam_beta1, wandb.config.adam_beta2),
    eps=wandb.config.adam_epsilon,
    weight_decay=wandb.config.weight_decay,
)
scheduler = OneCycleLR(
    optimizer, max_lr=wandb.config.learning_rate, total_steps=int(wandb.config.total_steps), last_epoch=-1
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
    compute_metrics=wandb_metric_computer(),
    **bookscorpusandwiki.trainer_params,
)

# run the training
trainer.train()

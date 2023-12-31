"""
Train script for any models. Set the configs using a wandb configuration .yaml file
by running `python train.py --configs configs/{config_file_here}.yaml`
"""

import wandb
from torch.optim.lr_scheduler import OneCycleLR

from src import SofterLlamaConfig, SofterLlamaForCausalLM
from src import SofterBertForMaskedLM
from src.dataloader import BooksCorpusAndWiki
from src.training_utils import SofterTrainer, SofterTrainingArguments
from src.quantization.quant import default_quant_configs

from transformers import LlamaTokenizerFast, BertTokenizer
from transformers.optimization import AdamW

# maps .yaml config file model names to model and tokenizer classes
model_mapping = {"softerllama": SofterLlamaForCausalLM, "softerbert": SofterBertForMaskedLM}
tokenizer_mapping = {"softerllama": LlamaTokenizerFast, "softerbert": BertTokenizer}

# the config files are loaded automatically into wandb.config using wandb --configs command line argument
wandb.init(project="training-runs", entity="softermax")
wandb.run.name = wandb.config.run_name
wandb.run.save()

# model configs setup
# TODO

# tokenizer setup
# TODO

# model setup
model = model_mapping[wandb.config.model_name](config=config)

# dataset setup
bookscorpusandwiki = BooksCorpusAndWiki(tokenizer, mlm_probability=wandb.config.mlm_probability)
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
    **bookscorpusandwiki.training_args,
    quant_kwargs=default_quant_configs,
)

trainer = SofterTrainer(model, training_args, optimizers=(optimizer, scheduler), **bookscorpusandwiki.trainer_params)

# run the training
trainer.train()

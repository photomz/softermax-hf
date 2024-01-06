"""
Train script for any models. Set the configs using a wandb configuration .yaml file
by running `python train.py --configs configs/{config_file_here}.yaml`
"""

import wandb
from torch.optim.lr_scheduler import OneCycleLR

from src import SofterLlamaConfig, SofterLlamaForCausalLM
from src import SofterBertConfig, SofterBertForMaskedLM
from src.dataloader import BooksCorpusAndWiki
from src.training_utils import SofterTrainer, SofterTrainingArguments, compute_softermetrics
from src.quantization.quant import default_quant_configs

from transformers import LlamaTokenizerFast, BertTokenizer
from transformers.optimization import AdamW

# maps .yaml config file model names to model and tokenizer classes
model_mapping = {"softerllama": SofterLlamaForCausalLM, "softerbert": SofterBertForMaskedLM}
config_mapping = {"softerllama": SofterLlamaConfig, "softerbert": SofterBertConfig}
tokenizer_mapping = {"softerllama": LlamaTokenizerFast, "softerbert": BertTokenizer}

# the config files are loaded automatically into wandb.config using wandb --configs command line argument
wandb.init(project="training-runs", entity="softermax")
wandb.run.name = wandb.config.run_name
wandb.run.save()

# model configs setup
config = config_mapping[wandb.config.model_name].from_pretrained(wandb.config.model_config_src)
config.n_bias = 1

# tokenizer setup
tokenizer = tokenizer_mapping[wandb.config.model_name].from_pretrained(wandb.config.model_config_src)

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
    output_dir=wandb.config.output_dir,
    max_steps=wandb.config.total_steps,
    evaluation_strategy="steps",
    eval_steps=wandb.config.eval_steps,
    eval_accumulation_steps=wandb.config.eval_accumulation_steps,
    save_strategy="steps",
    save_steps=wandb.config.save_steps,
    per_device_train_batch_size=wandb.config.batch_size,
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

# run the training
trainer.train()

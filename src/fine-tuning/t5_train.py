import json
import os

import datasets
import torch
import yaml
from torch.utils import data
from transformers import T5ForConditionalGeneration, T5Tokenizer, Trainer, TrainingArguments
from tqdm import tqdm

from utils.prompt import prompt_generation

os.environ["WANDB_PROJECT"] = "PRM"  # name your W&B project
os.environ["WANDB_LOG_MODEL"] = "end"  # log best model checkpoints
os.environ["WANDB_WATCH"] = "all"  # log gradients and parameters

with open('config.yml', 'r') as f:
    config = yaml.safe_load(f)

model_name = config['t5']
max_source_length = config['max_input_length']
max_target_length = config['max_output_length']

tokenizer = T5Tokenizer.from_pretrained(
                config['pretrained_model'].format(model=model_name))
model = T5ForConditionalGeneration.from_pretrained(
            config['pretrained_model'].format(model=model_name)).to('cuda')
train_file = os.path.join(config['tune_file_root'], config['train_file'])
dev_file = os.path.join(config['tune_file_root'], config['eval_file'])

class MDataset(data.Dataset):
    def __init__(self, data):
        self.input_ids = data['input_ids']
        self.attention_mask = data['attention_mask']
        self.labels = data['labels']

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {'input_ids': self.input_ids[idx],
                'attention_mask': self.attention_mask[idx],
                'labels': self.labels[idx]}

with open(train_file, 'r') as f:
    data = []
    for line in f.readlines():
        line = json.loads(line)
        data.append(line)

train_input = []
train_label = []
for d in data:
    prompt = prompt_generation(d, include_answer=True)
    x = prompt.split('Step 1')[0].strip()
    y = prompt[len(x):].strip()
    if '>>' in y:
        continue
    train_input.append(x)
    train_label.append(y)
tokenized_train_data = tokenizer(
        text=train_input,
        text_target=train_label,
        return_tensors='pt',
        padding=True,
        truncation=True,
        max_length=max_source_length
        )
train_dataset = MDataset(tokenized_train_data)

with open(dev_file, 'r') as f:
    data = []
    for line in f.readlines():
        line = json.loads(line)
        data.append(line)

dev_input = []
dev_label = []
for d in data:
    prompt = prompt_generation(d, include_answer=True)
    x = prompt.split('Step 1')[0].strip()
    y = prompt[len(x):].strip()
    if '>>' in y:
        continue
    dev_input.append(x)
    dev_label.append(y)
tokenized_dev_data = tokenizer(
        text=dev_input,
        text_target=dev_label,
        return_tensors='pt',
        padding=True,
        truncation=True,
        max_length=max_target_length
        )
dev_dataset = MDataset(tokenized_dev_data)

training_args = TrainingArguments(
    output_dir=config['checkpoints'].format(
                    model=model_name,
                    dataset=config['dataset']),
    logging_dir=config['logging'].format(
                    model=model_name,
                    dataset=config['dataset']),
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=5,
    save_steps=200,
    load_best_model_at_end=True,
    evaluation_strategy='steps',
    eval_steps=200,
    fp16=True,
    report_to='wandb',
    logging_steps=1,
    run_name="t5-large-prm800k-tune",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=dev_dataset,
)

trainer.train()

trainer.evaluate()

model.save_pretrained(config['tuned_model'].format(
                        model=model_name,
                        dataset=config['dataset']))
tokenizer.save_pretrained(config['tuned_model'].format(
                        model=model_name,
                        dataset=config['dataset']))


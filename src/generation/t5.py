import json
import os

import torch
import yaml
from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import GenerationConfig
from tqdm import tqdm

from utils.prompt import prompt_generation

with open('config.yml', 'r') as f:
    config = yaml.safe_load(f)

model_name = config['t5']
tokenizer = T5Tokenizer.from_pretrained(
                config['pretrained_model'].format(model=model_name))
model = T5ForConditionalGeneration.from_pretrained(
            config['tuned_model'].format(
                model=model_name,
                dataset=config['dataset']
            )).to('cuda')

input_file_name = os.path.join(config['dataset_root'],
                               config['sample_data'])
output_file_name = os.path.join(config['dataset_root'],
                                config['generation_file'].format(
                                    model=model_name))
k = config['num_of_runs_per_question']

with open(input_file_name, 'r') as f:
    data = json.load(f)

generation_config = GenerationConfig(
        do_sample=True,
        num_beams=5,
        num_return_sequences=5,
        early_stopping=True,
        temperature=0.8,
        max_new_tokens=config['max_output_length'],
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
        )

generations = {}

for d in tqdm(data):
    if 'musique' == config['dataset']:
        idx = d['id']
    elif 'prm800k' == config['dataset']:
        idx = d['timestamp']

    prompt = prompt_generation(d)
    
    input_ids = tokenizer(prompt, return_tensors="pt").to('cuda')
    outputs = model.generate(**input_ids, generation_config=generation_config).cpu()

    for i in range(k):
        output = tokenizer.decode(outputs[i], skip_special_tokens=True)
    
        if idx not in generations:
            generations[idx] = []
        generations[idx].append(output)

with open(output_file_name, 'w') as f:
    json.dump(generations, f, indent=2)


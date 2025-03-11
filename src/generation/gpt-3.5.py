import json
import os

import yaml
from tqdm import tqdm

from utils.openai_api import call
from utils.prompt import prompt_generation

with open('config.yml', 'r') as f:
    config = yaml.safe_load(f)

model_name = config['gpt-3.5']
demonstration_file_name = os.path.join(config['dataset_root'],
                                       config['demonstration'])
input_file_name = os.path.join(config['dataset_root'],
                               config['sample_data'])
output_file_name = os.path.join(config['dataset_root'],
                                config['generation_file'].format(
                                    model=model_name))
k = config['num_of_runs_per_question']

with open(demonstration_file_name, 'r') as f:
    demonstrations = json.load(f)

with open(input_file_name, 'r') as f:
    data = json.load(f)

generations = {}

demonstration_prompt = prompt_generation(demonstrations[1], include_answer=True)

for d in tqdm(data):
    if 'musique' == config['dataset']:
        idx = d['id']
    elif 'prm800k' == config['dataset']:
        idx = d['timestamp']

    for _ in range(k):
        prompt = prompt_generation(d)
        prompt = '\n\n'.join([demonstration_prompt, prompt])

        output = call(prompt, max_tokens=config['max_output_length'])
    
        if idx not in generations:
            generations[idx] = []
        generations[idx].append(output)

with open(output_file_name, 'w') as f:
    json.dump(generations, f, indent=2)


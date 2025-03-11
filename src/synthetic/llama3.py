import json
import os
import re

import yaml
from tqdm import tqdm

from utils.llama3_api import batch_call
from utils.hallucinaiton_prompt import prm800k

with open('config_math.yml', 'r') as f:
    config = yaml.safe_load(f)

model_name = config['llama3']
input_file_name = os.path.join(config['dataset_root'],
                               config['correct_file'])
output_file_name = os.path.join(config['dataset_root'],
                                config['synthetic_file'].format(
                                    hallucination='Reasoning-Step-Inconsistency',
                                    model=model_name))

data = []
with open(input_file_name, 'r') as f:
    for line in f:
        data.append(json.loads(line))
    data = data[:1000]

idxes = []
prompts = []
for d in tqdm(data):
    if 'musique' == config['dataset']:
        idx = d['id']
    elif 'prm800k' == config['dataset']:
        idx = d['timestamp']

    system_prompt, prompt = prm800k(d, 'Reasoning Step Inconsistency')

    idxes.append(idx)
    prompts.append((system_prompt, prompt))

outputs = batch_call(prompts, max_tokens=config['max_output_length'])
    
for idx, d, output in tqdm(zip(idxes, data, outputs)):
    match = re.search('\[Next Reasoning Step with.*\]\n(.*)$', output, re.M)
    if match is not None:
        output = match.group(1).strip()
    match = re.search('Step [\d]+:(.*)', output, re.I)
    if match is not None:
        output = match.group(1).strip()

    steps = d['label']['steps']
    s_idx = -1
    for i, step in enumerate(steps):
        completion = step['completions'][0]
        if -1 == completion['rating']:
            steps[i-1]['comletions'] = [{
                'text': output,
                'rating': 0,
                'flagged': None
            }]
            d['question']['pre_generated_steps'][i-1] = output
            s_idx = i
            break
    if -1 != s_idx:
        d['label']['steps'] = steps[:s_idx]
        d['question']['pre_generated_steps'] = d['question']['pre_generated_steps'][:s_idx]
    else:
        # if there is no hallucination in the golden reasoning chain, take the second last step since the last step is only a sentence to repeat the answer.
        d['label']['steps'] = d['label']['steps'][:-1]
        d['label']['steps'][-1]['completions'] = [{
            'text': output,
            'rating': 0,
            'flagged': None
        }]
        d['question']['pre_generated_steps'] = d['question']['pre_generated_steps'][:-1]
        d['question']['pre_generated_steps'][-1] = output

with open(output_file_name, 'w') as f:
    json.dump(data, f, indent=2)


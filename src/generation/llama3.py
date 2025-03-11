import json
import random
import re
import os

import yaml
from tqdm import tqdm

from utils.prompt import prompt_generation

with open('config.yml', 'r') as f:
    config = yaml.safe_load(f)

model_name = config['llama3-70b']
demonstration_file_name = os.path.join(config['dataset_root'],
                                       config['demonstration'])
input_file_name = os.path.join(config['dataset_root'],
                               config['sample_data'])
output_file_name = os.path.join(config['dataset_root'],
                                config['generation_file'].format(
                                    model=model_name))
file_name, ext = os.path.splitext(output_file_name)
intermediate_file_name = file_name + '_inter' + ext
k = config['num_of_runs_per_question']

with open(demonstration_file_name, 'r') as f:
    if config['dataset'] in ['math500', 'gsm8k']:
        demonstrations = []
        for line in f:
            demonstrations.append(json.loads(line))
    else:
        demonstrations = json.load(f)

if config['dataset'] in ['math500', 'gsm8k']:
    data = []
    with open(input_file_name, 'r') as file:
        # Read each line in the file
        for line in file:
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON: {e}")

else:
    with open(input_file_name, 'r') as f:
        data = json.load(f)
    
generations = {}

generated_data = {}
if os.path.isfile(intermediate_file_name):
    print('Intermediate File:', intermediate_file_name)
    inter_data = []
    with open(intermediate_file_name, 'r') as f:
        for line in f:
            inter_data.append(json.loads(line))
    
    for d in inter_data:
        d_inp = d['input']
        match = re.search(r'Question:([\S\s]*)\<\|eot_id\|\>', d_inp, re.M)
        if match is not None:
            question = match.group(1).strip()
        if question not in generated_data:
            generated_data[question] = []
        generated_data[question].append(d['output'])

print('Full Generations:', len(data), '*', k, '=', k * len(data))
print('Existing Generated Questions:', len(generated_data))

idxes = []
prompts = []
for d in tqdm(data):
    if 'musique' == config['dataset']:
        idx = d['id']
    elif 'prm800k' == config['dataset']:
        idx = d['timestamp']
    elif 'math500' == config['dataset']:
        idx = d['unique_id']
        question = d['problem'].strip()
    elif 'gsm8k' == config['dataset']:
        idx = d['id']
        question = d['question'].strip()

    # check intermediate data and skip generated data
    repeat_k = k
    if question in generated_data:
        if k == len(generated_data[question]):
            continue
        else:
            repeat_k = k - len(generated_data[question])

    for _ in range(repeat_k):
        prompt = prompt_generation(d)
#        prompt = '\n\n'.join([demonstration_prompt, prompt])
      
        idxes.append(idx)
        prompts.append(prompt)

print('Prepare to prompt:', len(prompts))

from utils.llama3_api import batch_call
outputs = batch_call(prompts, max_tokens=config['max_output_length'], batch_size=32, save_path=intermediate_file_name)

for idx, output in zip(idxes, outputs):
    if idx not in generations:
        generations[idx] = []
    generations[idx].append(output)

with open(output_file_name, 'w') as f:
    json.dump(generations, f, indent=2)


import json
import os

import yaml
from tqdm import tqdm

from utils.openai_api import call_batch, check_retrieve_batch
from utils.prompt import prompt_generation

with open('config.yml', 'r') as f:
    config = yaml.safe_load(f)

model_name = config['gpt-4']
demonstration_file_name = os.path.join(config['dataset_root'],
                                       config['demonstration'])
input_file_name = os.path.join(config['dataset_root'],
                               config['sample_data'])
output_file_name = os.path.join(config['dataset_root'],
                                config['generation_file'].format(
                                    model=model_name))
batch_id_file_name = os.path.join(config['dataset_root'],
                                  config['batch_id_file'])
batch_input_file_name = os.path.join(config['dataset_root'],
                                     config['batch_input_file'].format(
                                         model=model_name))
batch_output_file_name = os.path.join(config['dataset_root'],
                                     config['batch_output_file'].format(
                                         model=model_name))
k = config['num_of_runs_per_question']

with open(demonstration_file_name, 'r') as f:
    demonstrations = json.load(f)

with open(input_file_name, 'r') as f:
    data = json.load(f)

### generate prompt batch ###
demonstration_prompt = prompt_generation(demonstrations[1], include_answer=True)

batch_tasks = []
for d in tqdm(data):
    if 'musique' == config['dataset']:
        idx = d['id']
    elif 'prm800k' == config['dataset']:
        idx = d['timestamp']

    for try_idx in range(k):
        prompt = prompt_generation(d)
        prompt = '\n\n'.join([demonstration_prompt, prompt])

        task = {
            "custom_id": f"{idx}|||{try_idx}",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                # This is what you would have in your Chat Completions API call
                "model": model_name,
                "temperature": 0.8,
                "max_tokens": config['max_output_length'],
                "messages": [
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ],
            }
        }
        batch_tasks.append(task)

with open(batch_input_file_name, 'w') as f:
    for task in batch_tasks:
        f.write(json.dumps(task) + '\n')

### send batch to OpenAI ###
batch_id = call_batch(batch_input_file_name)
with open(batch_id_file_name, 'w') as f:
    f.write(batch_id)

### retrieve results from OpenAI ###
results = check_retrieve_batch(batch_id)
with open(batch_output_file_name, 'wb') as f:
    f.write(results)

results = []
with open(batch_output_file_name, 'r') as f:
    for line in f:
        json_object = json.loads(line.strip())
        results.append(json_object)

### extract results ###
generations = {}
for result in results:
    task_id = result['custom_id']
    data_idx, try_idx = task_id.split('|||')
    try_idx = int(try_idx)
    if data_idx not in generations:
        generations[data_idx] = [''] * k

    body = result['response']['body']
    if 'choices' in body:
        output = body['choices'][0]['message']['content']
    elif 'error' in body:
        output = body['error']['message']
    generations[data_idx][try_idx] = output

with open(output_file_name, 'w') as f:
    json.dump(generations, f, indent=2)


import json
import os

import yaml

from utils.openai_api import call_batch, check_retrieve_batch

with open('config.yml', 'r') as f:
    config = yaml.safe_load(f)

model_name = config['gpt-4']
output_file_name = os.path.join(config['dataset_root'],
                                config['generation_file'].format(
                                    model=model_name))
batch_id_file_name = os.path.join(config['dataset_root'],
                                  config['batch_id_file'])
batch_output_file_name = os.path.join(config['dataset_root'],
                                     config['batch_output_file'].format(
                                         model=model_name))
k = config['num_of_runs_per_question']

with open(batch_id_file_name, 'r') as f:
    batch_id = f.read().strip()

print(batch_id)

results = check_retrieve_batch(batch_id)
with open(batch_output_file_name, 'wb') as f:
    f.write(results)

results = []
with open(batch_output_file_name, 'r') as f:
    for line in f:
        json_object = json.loads(line.strip())
        results.append(json_object)

generations = {}
for result in results:
    task_id = result['custom_id']
    data_idx, try_idx = task_id.split('-')
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


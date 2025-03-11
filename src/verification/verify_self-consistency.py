import json
import random
import re

import yaml
from tqdm import tqdm

with open('config.yml', 'r') as f:
    config = yaml.safe_load(f)

model_name = config['llama3']
input_original_data = f'{config["dataset_root"]}{config["sample_data"]}'
input_generation_data = f'{config["dataset_root"]}generations_verify_{model_name}.json'
selected_generations_file = f'{config["dataset_root"]}selected_generations_sc.json'

with open(input_original_data, 'r') as f:
    data = []
    for line in f:
        data.append(json.loads(line))
    if 'math500' == config['dataset']:
        data_dict = {d['unique_id']:d for d in data}
    elif 'prm800k' == config['dataset']:
        data_dict = {d['timestamp']:d for d in data}
    elif 'gsm8k' == config['dataset']:
        data_dict = {d['id']:d for d in data}
    else:
        raise Error

error_answers = [str(-i*1.05) for i in range(10000, 20000)]
error_idx = 0

with open(input_generation_data, 'r') as f:
    generations = json.load(f)
    print(len(generations))

correct = 0
g_idx = 0
selected_generations = []
for k, v in tqdm(generations.items()):
    g_idx += 1
    d = data_dict[k]
    if 'math500' == config['dataset']:
        ground_truth_answer = d['answer']
    elif 'prm800k' == config['dataset']:
        ground_truth_answer = d['question']['ground_truth_answer']
    elif 'gsm8k' == config['dataset']:
        ground_truth_answer = d['answer'][0]
    else:
        raise Error

    v = random.sample(v, k=config['scaling_num_of_runs'])

    match_idx = -1
    all_candidate_answers = []
    for a_idx, answer in enumerate(v):
        matches = re.search('# Answer[\\n]*(.*)$', answer, re.M)
        if matches is not None:
            all_candidate_answers.append(matches.group(1).strip())
        else:
            print(answer)
            candidate = error_answers[error_idx]
            error_idx += 1
            all_candidate_answers.append(candidate)

    selected = max(set(all_candidate_answers), key=all_candidate_answers.count)

    print(g_idx, '-', ground_truth_answer, ',', selected)
    if ground_truth_answer in selected:
        correct += 1

with open(selected_generations_file, 'w') as f:
    for g in selected_generations:
        f.write(json.dumps(g)+'\n')

print(correct)
print(correct / len(generations))


import json
import re

import yaml
from tqdm import tqdm

with open('config.yml', 'r') as f:
    config = yaml.safe_load(f)

model_name = config['llama3']
input_original_data = '../data/prm800k_ori/prm800k/data/phase2_test.jsonl'
input_generation_data = '../data/prm800k_sample/generations_{}.json'.format(model_name)

with open(input_original_data, 'r') as f:
    data = []
    for line in f:
        data.append(json.loads(line))
    data_dict = {d['timestamp']:d for d in data}

with open(input_generation_data, 'r') as f:
    generations = json.load(f)

correct = 0
g_idx = 0
for k, v in tqdm(generations.items()):
    g_idx += 1
    d = data_dict[k]
    ground_truth_answer = d['question']['ground_truth_answer']

    match_idx = -1
    for a_idx, answer in enumerate(v):
        matches = re.search('# Answer[\\n]*(.*)$', answer, re.M)
        if matches is not None:
            selected = matches.group(1).strip()
        else:
            continue

        if ground_truth_answer in selected:
            print(g_idx, '-', ground_truth_answer, '-', selected)
            correct += 1
            break

print(correct)
print(correct / len(generations))


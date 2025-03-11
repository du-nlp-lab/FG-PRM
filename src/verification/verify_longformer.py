import json
import math
import os
import random
import re

import accelerate
import numpy as np
import torch
import torch.nn.functional as F
import transformers
import accelerate
import wandb
import yaml
import nltk
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from my_longformer import LongformerForTokenClassification
from verification.reward import FactualityReward
from utils.prompt import prompt_generation, answer_generation

with open('config.yml', 'r') as f:
    config = yaml.safe_load(f)

model_name = config['llama3-70b']
input_original_data = f'{config["dataset_root"]}{config["sample_data"]}'
input_generation_data = f'{config["dataset_root"]}generations_verify_{model_name}_64.json'
selected_generations_file = f'{config["dataset_root"]}selected_generations_longformer_fgprm.json'
reward_raw_score_path = f'{config["dataset_root"]}reward_raw_score_longformer.json'

reward_model_names = [
#    'ori',
    'Context-Inconsistency',
    'Logical-Inconsistency',
    'Instruction-Inconsistency',
    'Factual-Inconsistency',
    'Fabrication',
    'Calculation-Error',
]

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

with open(input_generation_data, 'r') as f:
    generations = json.load(f)
    print(len(generations))

tokenizer = transformers.AutoTokenizer.from_pretrained(
                '../models/t5-large-1k-train',
                model_max_length=1526
            )
tokenizer.padding_side = 'right'
tokenizer.max_input_len = 1526
tokenizer.max_generated_len = 200
 
reward_results = {}
for reward_name in reward_model_names:
    print(reward_name)
    accelerator = accelerate.Accelerator()
    reward_model = FactualityReward(
        tokenizer,
        f'../models/orm_longformer_{reward_name}',
        0.5,
        -0.5,
        sep = '</s>'
    )
    reward_model.f_reward_model = accelerator.prepare(reward_model.f_reward_model)
    
    for k, v in tqdm(generations.items()):
        d = data_dict[k]
        question = prompt_generation(d).split('Step 1')[0].strip()
    
        if k not in reward_results:
            reward_results[k] = []

        v = random.sample(v, k=config['scaling_num_of_runs'])

        for a_idx, answer in enumerate(v):
            answer_tok = tokenizer(
                [answer],
                return_tensors='pt',
                truncation=True,
                max_length=config['max_output_length'],
            )
            generated_input_ids = answer_tok.input_ids
            generated_attention_mask = answer_tok.attention_mask
            generated_texts = [answer]
            metadata = [{'prompt': question}]
        
            reward = reward_model.get_reward(
                torch.rand(2,2), torch.rand(2,2), 
                generated_input_ids, generated_attention_mask, 
                generated_texts, metadata
            )
            
            n_sentences = reward['n_sentences']
            n_factuality_correct = reward['n_corrects']
            raw_rewards = reward['raw_rewards'][0]
            raw_rewards = torch.stack(raw_rewards, dim=0)
        
       
            raw_rewards = F.log_softmax(raw_rewards[:, [0, 2]], dim=1)
            reward_score = torch.sum(raw_rewards[:, -1]).item()
    
            if a_idx >= len(reward_results[k]):
                reward_results[k].append([reward_score])
            else:
                reward_results[k][a_idx].append(reward_score)

    del accelerator
    del reward_model

print(reward_results)
with open(reward_raw_score_path, 'w') as f:
    json.dump(reward_results, f, indent=2)

final_results = {}
correct = 0
g_idx = 0
selected_generations = []
for k, reward_scores in reward_results.items():
    g_idx += 1
    max_score = -math.inf
    max_idx = -1
    for idx, scores in enumerate(reward_scores):
        final_score = np.sum(scores)
        if final_score > max_score:
            max_score = final_score
            max_idx = idx
    final_results[k] = max_idx
    
    d = data_dict[k]
    if 'math500' == config['dataset']:
        ground_truth_answer = d['answer'].strip()
    elif 'prm800k' == config['dataset']:
        ground_truth_answer = d['question']['ground_truth_answer'].strip()
    elif 'gsm8k' == config['dataset']:
        ground_truth_answer = d['answer'][0].strip()
    else:
        raise Error

    selected = generations[k][max_idx]
    selected_generations.append(selected)
    matches = re.search('# Answer[\\n]*(.*)$', selected, re.M)
    if matches is not None:
        selected = matches.group(1).strip()
    print(g_idx, '-', ground_truth_answer, ',', selected, ',', ground_truth_answer in selected)
    if ground_truth_answer in selected:
        correct += 1

with open(selected_generations_file, 'w') as f:
    for g in selected_generations:
        f.write(json.dumps(g)+'\n')

print(correct)
print(correct / len(generations))


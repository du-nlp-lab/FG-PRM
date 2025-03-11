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

from my_longformer import LongformerForTokenClassification
from verification.reward import FactualityReward
from utils.prompt import prompt_generation, answer_generation

with open('config.yml', 'r') as f:
    config = yaml.safe_load(f)

model_name = config['llama3-70b']
input_original_data = f'{config["dataset_root"]}{config["sample_data"]}'
input_generation_data = f'{config["dataset_root"]}generations_verify_{model_name}_64.json'
selected_generations_file = f'{config["dataset_root"]}selected_generations_longformer_single_orm.json'

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
 
accelerator = accelerate.Accelerator()
factuality_reward = FactualityReward(
    tokenizer,
#    '../models/prm_longformer_ori',
    '../models/prm_longformer_cg',
    0.5,
    -0.5,
    sep = '</s>'
)
factuality_reward.f_reward_model = accelerator.prepare(factuality_reward.f_reward_model)

correct = 0
idx = 0
selected_generations = []
for k, v in generations.items():
    idx += 1
    d = data_dict[k]
    question = prompt_generation(d).split('Step 1')[0].strip()

    v = random.sample(v, k=config['scaling_num_of_runs'])

    max_score = -math.inf
    max_idx = None
    for a_idx, answer in enumerate(v):
   
        answer_tok = tokenizer(
                            [answer],
                            return_tensors='pt',
                            truncation=True,
                            max_length=config['max_input_length'],
                            )
        generated_input_ids = answer_tok.input_ids
        generated_attention_mask = answer_tok.attention_mask
        generated_texts = [answer]
        metadata = [{'prompt': question}]
    
        factuality = factuality_reward.get_reward(
                        torch.rand(2,2), torch.rand(2,2), 
                        generated_input_ids, generated_attention_mask, 
                        generated_texts, metadata)
        
        n_sentences = factuality['n_sentences']
        n_factuality_correct = factuality['n_corrects']
        factuality_rewards = factuality['factuality_rewards'][0]
        raw_rewards = factuality['raw_rewards'][0]
        raw_rewards = torch.stack(raw_rewards, dim=0)
    
        raw_rewards = F.log_softmax(raw_rewards, dim=1)
        reward_score = torch.sum(raw_rewards[:, -1]).item()
        if reward_score > max_score:
            max_score = reward_score
            max_idx = a_idx

    if 'math500' == config['dataset']:
        ground_truth_answer = d['answer'].strip()
    elif 'prm800k' == config['dataset']:
        ground_truth_answer = d['question']['ground_truth_answer'].strip()
    elif 'gsm8k' == config['dataset']:
        ground_truth_answer = d['answer'][0].strip()
    else:
        raise Error

    selected = v[max_idx]
    selected_generations.append(selected)
    matches = re.search('# Answer[\\n]*(.*)$', selected, re.M)
    if matches is not None:
        selected = matches.group(1).strip()
    print(idx, '-', ground_truth_answer, ',', selected, ',', ground_truth_answer in selected)
    if ground_truth_answer in selected:
        correct += 1

with open(selected_generations_file, 'w') as f:
    for g in selected_generations:
        f.write(json.dumps(g)+'\n')

print(correct)
print(correct / len(generations))


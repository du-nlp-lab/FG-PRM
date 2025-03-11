import json
import math
import os
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
from sklearn.metrics import precision_recall_fscore_support

from my_longformer import LongformerForTokenClassification
from verification.reward import FactualityReward
from utils.prompt import prompt_generation, answer_generation

with open('config.yml', 'r') as f:
    config = yaml.safe_load(f)

hallucinations = [
#    'ori_train',
#    'ori_test',
    'Calculation-Error',
    'Logical-Inconsistency',
    'Context-Inconsistency',
    'Instruction-Inconsistency',
    'Factual-Inconsistency',
    'Fabrication',
]
hallucination = hallucinations[0]

model_name = config['llama3']
input_original_data = f'../pata/hallucination_sample/synthetic_{hallucination}_step-injected.json'

data = []
with open(input_original_data, 'r') as f:
    for line in f:
        data.append(json.loads(line))
    data_dict = {d['timestamp']:d for d in data}

tokenizer = transformers.AutoTokenizer.from_pretrained(
                '../models/t5-large-1k-train',
                model_max_length=1024
            )
tokenizer.padding_side = 'right'
tokenizer.max_input_len = 1024
tokenizer.max_generated_len = 200
 
accelerator = accelerate.Accelerator()
factuality_reward = FactualityReward(
    tokenizer,
    '../models/orm_longformer_ori',
    0.5,
    -0.5,
    sep = '</s>'
)
factuality_reward.f_reward_model = accelerator.prepare(factuality_reward.f_reward_model)

all_prec, all_recall, all_fscore = 0, 0, 0
last_correct = 0
for d in data:
    question = d['question']['problem']

    answer_text, answer_label = [], []
    steps = d['label']['steps']
    for step in steps:
        step_text = options[0]['text'].strip()
        answer_text.append(step_text + '</s>')

        label = options[0]['rating']
        if -1 == label:
            answer_label.append(0)
            break
        else:
            answer_label.append(1)

    remaining_steps = d['question']['pre_generated_steps'][len(answer):]
    for step in remaining_steps:
        answer_text.append(step.strip())

    answer_text = ' '.join(answer_text)
    answer_tok = tokenizer(
                        [answer_text],
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
    raw_rewards = torch.stack(raw_rewards, dim=0)[:, [0,2]]
    results = torch.amax(raw_rewards, dim=1).tolist()

    assert len(answer_label) == len(results)
    last_correct += (answer_label[-1] == results[-1])
    all_golden.extend(answer_label)
    all_pred.extend(results)
    
    prec, recall, fscore, _ = precision_recall_fscore_support(answer_label, results)
    all_prec += prec
    all_recall += recall
    all_fscore += fscore

print(all_prec/len(data))
print(all_recall/len(data))
print(all_fscore/len(data))


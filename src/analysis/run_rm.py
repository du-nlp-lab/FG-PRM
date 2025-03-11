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
from tqdm import tqdm

from verification.reward import FactualityReward
from utils.prompt import prompt_generation, answer_generation

with open('config.yml', 'r') as f:
    config = yaml.safe_load(f)

model_name = config['llama3']

tokenizer = transformers.AutoTokenizer.from_pretrained(
                '../models/t5-large-1k-train',
                model_max_length=1526
            )
tokenizer.padding_side = 'right'
tokenizer.max_input_len = 1526
tokenizer.max_generated_len = 200
 
reward_results = {}

def run_rm(prompts, hallucination_type, save_path=None):

    accelerator = accelerate.Accelerator()
    reward_model = FactualityReward(
        tokenizer,
        f'../models/orm_longformer_{hallucination_type}',
#        f'../models/prm_longformer_sixinone',
#        f'../models/prm_longformer_ori',
        0.5,
        -0.5,
        sep = '</s>'
    )
    reward_model.f_reward_model = accelerator.prepare(reward_model.f_reward_model)

    if save_path is not None:
        fp = open(save_path, 'w')

    responses = []
    for prompt in tqdm(prompts):
        question, rs = prompt

        question = question.strip()

        answer_tok = tokenizer(
            [rs],
            return_tensors='pt',
            truncation=True,
            max_length=config['max_output_length'],
        )
        generated_input_ids = answer_tok.input_ids
        generated_attention_mask = answer_tok.attention_mask
        generated_texts = [rs]
        metadata = [{'prompt': question}]

        reward = reward_model.get_reward(
                        torch.rand(2,2), torch.rand(2,2), 
                        generated_input_ids, generated_attention_mask, 
                        generated_texts, metadata)
        
        raw_rewards = reward['raw_rewards'][0]
        raw_rewards = torch.stack(raw_rewards, dim=0)
     
        raw_rewards = torch.argmax(raw_rewards, dim=1)
        raw_rewards = raw_rewards.tolist()
        response = []
        for r_idx, reward in enumerate(raw_rewards):
            text = 'Step {}: {}'.format(
                r_idx,
                'Yes' if 6 != reward else 'No'
            )
            response.append(text)
        response = '\n'.join(response)
        responses.append(response)

        if save_path is not None:
            fp.write(json.dumps({'input': prompt, 'output': response})+'\n')

    if save_path is not None:
        fp.close()

    return responses


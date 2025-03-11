import json
import os
import sys

import pandas as pd
import yaml
from tqdm import tqdm

from utils.openai_api import call_chat
from utils.llama3_api import batch_call
from utils.hallucination_prompt import generate_hallucination_prompts #, generate_hallucination_prompts_from_file

with open('config.yml', 'r') as f:
    config = yaml.safe_load(f)

model_list = ['llama3']
#model_list = ['gpt-3.5']
#model_list = ['gpt-4']

for model in model_list:
    model_name = config[model]

    hallucination_type = [
        'Fabrication',
        'Factual Inconsistency',
        'Context Inconsistency',
        'Instruction Inconsistency',
        'Logical Inconsistency',
        'Calculation Error',
    ]
    for hallucination in hallucination_type:
        judge_available_file_name = os.path.join(config['dataset_root'], model_name + '_' + hallucination.replace(' ', '-') + '_flags.json')
        # prompt generation
        dataset, sys_prompts, prompts = generate_hallucination_prompts(
                hallucination,
                dataset='gsm8k',
                n_samples=700,
                save_path=judge_available_file_name
                )
#        dataset, sys_prompts, prompts = generate_hallucination_prompts_from_file(
#                hallucination_type=hallucination,
#                dataset=data,
#                n_samples=-1,
#        )
        if not isinstance(dataset, pd.DataFrame):
            dataset = dataset.to_pandas()
        dataset['system_prompt'] = sys_prompts
        dataset['prompt'] = prompts

        # save prompts
        output_file_name = os.path.join(config['dataset_root'], model_name + '_' + hallucination.replace(' ', '-') + '.json')
        dataset.to_json(output_file_name, orient='records', lines=True)

        # llama3 generation
        if 'llama-' == model_name[:6]:
            all_prompts = list(zip(sys_prompts, prompts))
            outputs = batch_call(all_prompts,
                                 max_tokens=config['max_output_length'],
                                 batch_size=4,
                                 save_path=os.path.splitext(output_file_name)[0]+'_intermediate.json'
                                 )
        elif 'gpt-' == model_name[:4]:
            outputs = []
            for sys_prompt, prompt in tqdm(zip(sys_prompts, prompts)):
                messages = [{'role': 'system', 'content': sys_prompt},
                            {'role': 'user', 'content': prompt}]
                outputs.append(call_chat(messages,
                                         model=model_name,
                                         max_tokens=config['max_output_length']))

        # save outputs
        dataset['generated_cot'] = outputs
        dataset.to_json(output_file_name, orient='records', lines=True)


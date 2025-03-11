import json
import re
import os
from statistics import fmean

import yaml
from sklearn.metrics import f1_score, precision_recall_fscore_support

from utils.openai_api import call_batch, check_retrieve_batch

hallucinations = [
    'Calculation-Error',
    'Logical-Inconsistency',
    'Context-Inconsistency',
    'Instruction-Inconsistency',
    'Factual-Inconsistency',
    'Fabrication',
]

with open('config.yml', 'r') as f:
    config = yaml.safe_load(f)

def extract_results(prompts, responses):
    data = []
    for prompt, response in zip(prompts, responses):
        new_data = {'prompt': prompt, 'response': response}

        match = re.search('(?s:.*)step 1:', response, re.I)
        if match is not None:
            span = match.span()
            response = response[span[-1]-7:].strip()

        response = [r.strip() for r in response.split('\n') if '' != r]
        flags = []
        for step in response:
            match = re.search(r'Step (\d): (Yes|No)', step)
            if match is not None: 
                idx = int(match.group(1))
                content = match.group(2)
                flag = 1 if 'Yes'==content else 0
                if len(flags) >= idx:
                    flags[idx-1] = flag
                else:
                    flags.append(flag)
        new_data['flags'] = flags
        data.append(new_data)

    return data

def calculate_f1_scores(data, results):
    precision_scores, recall_scores, f1_scores = [], [], []
    for d, r in zip(data, results):
        golden = [0] * len(d['cot'])
        golden[d['inject_step']-1] = 1

        flags = r['flags']

        if len(golden) != len(flags):
            flags = flags[:len(golden)]

        precision, recall, fscore, _ = precision_recall_fscore_support(golden, flags, average='binary')
        precision_scores.append(precision)
        recall_scores.append(recall)
        f1_scores.append(fscore)

    return fmean(precision_scores), fmean(recall_scores), fmean(f1_scores)

def binary_count(all_results):
    count = []
    for k, v in all_results.items:
        for v_idx, d in enumerate(v):
            if v_idx >= len(count):
                count.append([])
            count[v_idx].append(int(1 in d['flags']))

    cnt = 0
    for c in count:
        if 1 in c:
            cnt += 1
    print(cnt / len(count))

all_results = {}
for hallucination in hallucinations:

    ### read original sample data ###
    data = []
    data_file_path = '{root}{file_name}'.format(
        root=config['dataset_root'],
        file_name=config['evaluation_sample_file'].format(
            hallucination=hallucination,
            number=config['sample_size'],
        )
    )
    with open(data_file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))

    ### retrieve gpt batch results ###
    model_name = config[config['evaluation_model']]

    batch_id_file_name = os.path.join(
        config['evaluation_root'],
        config['evaluation_batch_id_file'].format(
            dataset=config['dataset'],
            hallucination=hallucination,
            model=model_name
        )
    )
    with open(batch_id_file_name, 'r') as f:
        batch_id = f.read().strip()
    
    print(batch_id)

    batch_output_file_name = os.path.join(
        config['evaluation_root'],
        config['evaluation_batch_output_file'].format(
            dataset=config['dataset'],
            hallucination=hallucination,
            model=model_name
        )
    )
    outputs = check_retrieve_batch(batch_id)
    with open(batch_output_file_name, 'wb') as f:
        f.write(outputs)

    outputs = []
    with open(batch_output_file_name, 'r') as f:
        for line in f:
            outputs.append(json.loads(line.strip()))
    outputs_dict = {}
    for output in outputs:
        task_id = int(output['custom_id'])

        body = output['response']['body']
        if 'choices' in body:
            output = body['choices'][0]['message']['content']
        elif 'error' in body:
            output = body['error']['message']

        outputs_dict[task_id] = output
    responses = [v for k, v in sorted(outputs_dict.items())]
    
    ### retrieve prompts ###
    # run prompt-based evaluation
    intermediate_file_path = '{root}{file_name}'.format(
        root = config['evaluation_root'],
        file_name = config['evaluation_intermediate_file'].format(
            dataset=config['dataset'],
            hallucination=hallucination,
            model=config[config['evaluation_model']]
        )
    )
    intermediate_results = []
    with open(intermediate_file_path, 'r') as f:
        for line in f:
            intermediate_results.append(json.loads(line))
    prompts = []
    for result in intermediate_results:
        prompts.append(result['input'])

    ### extract results ###
    result_file_path = '{root}{file_name}'.format(
        root = config['evaluation_root'],
        file_name = config['evaluation_result_file'].format(
            dataset=config['dataset'],
            hallucination=hallucination,
            model=config[config['evaluation_model']]
        )
    )
    if False: #os.path.isfile(result_file_path):
        results = []
        with open(result_file_path, 'r') as f:
            for line in f:
                results.append(json.loads(line))
    else:
        results = extract_results(prompts, responses)
        print(results)
        with open(result_file_path, 'w') as f:
            for r in results:
                f.write(json.dumps(r)+'\n')
    all_results[hallucination] = results

    # compare results with data to get F1 scores
    precision, recall, f1_score = calculate_f1_scores(data, results)
    print(hallucination, 'precision:', precision)
    print(hallucination, 'recall:', recall)
    print(hallucination, 'f1:', f1_score)

binary_count(results)


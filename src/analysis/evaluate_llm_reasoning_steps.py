import json
import random
import re
import os
from statistics import fmean

import yaml
from sklearn.metrics import f1_score, precision_recall_fscore_support
from tqdm import tqdm

with open('config.yml', 'r') as f:
    config = yaml.safe_load(f)

hallucinations = [
    'Calculation-Error',
    'Logical-Inconsistency',
    'Context-Inconsistency',
    'Instruction-Inconsistency',
    'Factual-Inconsistency',
    'Fabrication',
]

rm_prompt = (
'Please answer the following math question step-by-step. The output should '
'follow the rules:\n'
'1. Each step should be in a new line and start with "Step <number>: " '
'indicating the step index.\n'
'2. If the final answer is deduced, please follow the format, '
'"# Answer\\n\\n<answer>", to output your final answer individually '
'in the last line.\n'
'3. If the number of steps is larger than 50 or reasoning steps ends with '
'no proper answer, please only output "No Answer" in the last line.\n\n'
'Questions: {question}'
)

def answer_questions(questions, save_path, model_name):
    prompts = []
    for question in questions:
        prompt = rm_prompt.format(question=question) + '\n\nAnswer:\n'
        prompts.append(prompt)

    if 'prm_model' == model_name:
        import analysis.run_rm as reward_model
        responses = reward_model.run_rm(prompts, hallucination_type, save_path)
    elif 'llama3' == model_name:
        import utils.llama3_api as llama3_api
        responses = llama3_api.batch_call(
            prompts,
            max_tokens=500,
            batch_size=16,
            save_path=save_path
        )
    elif 'claude' == model_name[:6]:
        import utils.anthropic_api as anthropic_api
        responses = []
        if save_path is not None:
            fp = open(save_path, 'w')
        for prompt in tqdm(prompts):
            messages = [{'role': 'user', 'content': prompt}]
            response = anthropic_api.call(messages, max_tokens=config['max_output_length'])
            responses.append(response)
            if save_path is not None:
                fp.write(json.dumps({'input': prompt, 'output': response})+'\n')
        if save_path is not None:
            fp.close()
    elif 'gpt' == model_name[:3] or 'o1' == model_name[:2]:
        import utils.openai_api as openai_api

        ### normal version ###
        responses = []
        if save_path is not None:
            fp = open(save_path, 'w')
        for prompt in tqdm(prompts):
            response = openai_api.call(prompt, max_tokens=config['max_output_length'])
            responses.append(response)
            if save_path is not None:
                fp.write(json.dumps({'input': prompt, 'output': response})+'\n')
        if save_path is not None:
            fp.close()

    return prompts, responses

def extract_results(prompts, responses):
    data = []
    for prompt, response in tqdm(zip(prompts, responses)):
        new_data = {'prompt': prompt, 'response': response}

        match = re.search('(?s:.*)(step 1:)( )+(yes|no)', response, re.I)
        if match is not None:
            span = match.span(1)
            response = response[span[-1]-7:].strip()

        response = [r.strip() for r in response.split('\n') if '' != r]
        flags = []
        for step in response:
            match = re.search(r'Step (\d): (Yes|No)', step)
            if match is not None: 
                idx = int(match.group(1))
                content = match.group(2)
                flag = 1 if 'Yes'==content else 0
                if len(flags) > idx:
                    flags[idx] = flag
                else:
                    flags.append(flag)
        new_data['flags'] = flags
        data.append(new_data)

    return data

def calculate_hallucination_score(results):
    scores = []
    for r in results:
        r = r['flags']
        scores.append(sum(r) / len(r))
    avg_score = sum(scores) / len(results)
    return avg_score

if '__main__' == __name__:
    # read data
    all_data_dict = {}
    model_name = config['llama3-70b']
    data_file_path = f'{config["dataset_root"]}MATH500.jsonl'
    with open(data_file_path, 'r') as f:
        for line in f:
            instance = json.loads(line)
            all_data_dict[instance['unique_id']] = instance

    questions = []
    data = []
    generations_file = f'{config["dataset_root"]}generations_verify_{model_name}_64.json'
    with open(generations_file, 'r') as f:
        generations = json.load(f)
    for k in generations.keys():
        questions.append(all_data_dict[k]['problem'])
        data.append(all_data_dict[k])

    if 'prm_model' != config['evaluation_model']:
        # generate new cot generation
        intermediate_file_path = '{root}{file_name}'.format(
            root = config['evaluation_root'],
            file_name = config['evaluation_llm_intermediate_file'].format(
                dataset=config['dataset'],
                model=config[config['evaluation_model']],
                method=''
            )
        )
        print(intermediate_file_path)
        if os.path.isfile(intermediate_file_path): # False: #
            intermediate_results = []
            with open(intermediate_file_path, 'r') as f:
                for line in f:
                    intermediate_results.append(json.loads(line))
            prompts, responses = [], []
            for result in intermediate_results:
                prompts.append(result['input'])
                responses.append(result['output'])
        else:
            print('generate new cot answers')
            reasoning_steps = []
            prompts, responses = answer_questions(
                    questions,
                    intermediate_file_path,
                    model_name=config['evaluation_model']
            )
    else:
        model_name = config['llama3-70b']
        generations_file_path = f'{config["dataset_root"]}generations_verify_{model_name}_64.json'
        with open(generations_file_path, 'r') as f:
            generations = json.load(f)
            print(len(generations))

        # generate new cot generation
        intermediate_file_path = '{root}{file_name}'.format(
            root = config['evaluation_root'],
            file_name = config['evaluation_llm_intermediate_file'].format(
                dataset=config['dataset'],
                model=config[config['evaluation_model']],
                method=config['evaluation_method']
            )
        )

        if 'sc' == config['evaluation_method']:
            error_answers = [str(-i*1.05) for i in range(10000, 20000)]
            error_idx = 0

            responses = []
            for k, v in tqdm(generations.items()):
                match_idx = -1
                all_candidate_answers = []
                for a_idx, answer in enumerate(v):
                    matches = re.search('# Answer[\\n]*(.*)$', answer, re.M)
                    if matches is not None:
                        all_candidate_answers.append(matches.group(1).strip())
                    else:
                        candidate = error_answers[error_idx]
                        error_idx += 1
                        all_candidate_answers.append(candidate)

                selected = max(set(all_candidate_answers), key=all_candidate_answers.count)

                possible_cots = []
                for a_idx, answer in enumerate(v):
                    matches = re.search('# Answer[\\n]*(.*)$', answer, re.M)
                    if matches is not None:
                        possible_cots.append(answer)
                
                selected_cot = random.choice(possible_cots)
                responses.append(selected_cot)
        elif 'orm' == config['evaluation_method']:
            responses = []
            file_name = f'{config["dataset_root"]}selected_generations_longformer_single_orm.json'
            with open(file_name, 'r') as f:
                for line in f:
                    responses.append(json.loads(line))
        elif 'prm' == config['evaluation_method']:
            responses = []
            file_name = f'{config["dataset_root"]}selected_generations_longformer_single_prm.json'
            with open(file_name, 'r') as f:
                for line in f:
                    responses.append(json.loads(line))
        elif 'fg-prm' == config['evaluation_method']:
            responses = []
            file_name = f'{config["dataset_root"]}selected_generations_longformer_fgprm.json'
            with open(file_name, 'r') as f:
                for line in f:
                    responses.append(json.loads(line))

    for hallucination in hallucinations:
        # extract results
        result_file_path = '{root}{file_name}'.format(
            root = config['evaluation_root'],
            file_name = config['evaluation_llm_result_file'].format(
                dataset=config['dataset'],
                hallucination=hallucination,
                model=config[config['evaluation_model']],
                method='_'+config['evaluation_method']
            )
        )
        if False: # os.path.isfile(result_file_path): #
            results = []
            with open(result_file_path, 'r') as f:
                for line in f:
                    results.append(json.loads(line))
        else:
            print('extract results')
            # call rm to get results
            prompts = []
            for question, rs in zip(questions, responses):
                if 'No Answer' == rs:
                    rs = 'Step 1: No Answer'
                prompt = (question, rs)
                prompts.append(prompt)
    
            import analysis.run_rm as reward_model
            responses = reward_model.run_rm(prompts, hallucination)

            results = extract_results(prompts, responses)
            with open(result_file_path, 'w') as f:
                for r in results:
                    f.write(json.dumps(r)+'\n')

        score = calculate_hallucination_score(results)
        print(hallucination, score)


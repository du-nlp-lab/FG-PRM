import yaml
import json
import random
import re
with open('config.yml', 'r') as f:
    config = yaml.safe_load(f)

def prompt_generation(d, include_answer=False, evaluate=False):
    if 'musique' == config['dataset']:
        return prompt_generation_musique(d, include_answer, evaluate)
    elif 'prm800k' == config['dataset']:
        return prompt_generation_prm800k(d, include_answer, evaluate)
    elif 'math500' == config['dataset']:
        return prompt_generation_math500(d, include_answer, evaluate)
    elif 'math-shepherd' == config['dataset']:
        return prompt_generation_prm800k(d, include_answer, evaluate)
    elif 'gsm8k' == config['dataset']:
        return prompt_generation_gsm8k(d, include_answer, evaluate)

def answer_generation(d, include_answer=False, evaluate=False):
    if 'musique' == config['dataset']:
        return answer_generation_musique(d)
    elif 'prm800k' == config['dataset']:
        return answer_generation_prm800k(d)
    elif 'math500' == config['dataset']:
        return answer_generation_math500(d)
    elif 'gsm8k' == config['dataset']:
        return answer_generation_gsm8k(d)
    
### For Musique ###
def prompt_generation_musique(d, include_answer=False, evaluate=False):
    paragraphs_dict = {}
    for p in d['paragraphs']:
        paragraphs_dict[p['idx']] = p

    golden_documents = []
    golden_documents_idxes = set()
    unanswerable_flag = False
    for i, sq in enumerate(d['question_decomposition']):
        idx = sq['paragraph_support_idx']
        if idx is None or idx not in paragraphs_dict:
            unanswerable_flag = True
            continue
        golden_documents_idxes.add(idx)
        p = paragraphs_dict[idx]
        golden_document = 'Document {}: {}\n{}'.format(
                i+1, p['title'], p['paragraph_text'])
        golden_documents.append(golden_document)
    if unanswerable_flag and 5 > len(golden_documents):
        # add random documents for unanswerable questions
        all_keys = list(paragraphs_dict.keys())
        random.shuffle(all_keys)
        for idx in all_keys:
            if idx in golden_documents_idxes:
                continue
            p = paragraphs_dict[idx]
            golden_document = 'Document {}: {}\n{}'.format(
                    len(golden_documents)+1, p['title'], p['paragraph_text'])
            golden_documents.append(golden_document)
            if 5 <= len(golden_documents):
                break
    golden_documents = '\n\n'.join(golden_documents)

    prompt = (
        'Please answer the following question step-by-step according to the above documents. The output should follow the rules:\n'
        '1. Each step should be in question answering format as demonstrated.\n' # reasoning process.\n'
        '2. Each step should be in a new line and start with "Step <number>: " indicating the step index.\n'
        '3. If the final answer is deduced, please follow the format, "Answer: <answer>", to output your final answer individually '
        'in the last line.\n'
        '4. If the number of steps is larger than 10 or reasoning steps ends with no proper answer, please only output '
        '"No Answer" in the last line.'
              )

    question = 'Question: {}'.format(d['question'])
    # Ensure that your response is directly related to the document content and addresses all aspects of the question.
    prompt = f"""
Please provide a detailed and accurate answer to the question based on the information provided in the document below.
 
Question: {d['question']}

{golden_documents}
    """
    if evaluate:
        prompt = '\n\n'.join([golden_documents, question])
        return prompt

    if include_answer:
        answer = answer_generation(d)
        prompt = '\n\n'.join([golden_documents, prompt, question, answer])

    return prompt

def answer_generation_musique(d):
    qa_pairs = []
    for idx, qa in enumerate(d['question_decomposition']):
        qa_pair = 'Step {}: '.format(idx + 1) + qa['question'].capitalize()
        qa_pair = qa_pair.strip()
        if '?' != qa_pair[-1]:
            qa_pair = qa_pair + '? '
        else:
            qa_pair = qa_pair[:-1].strip() + '?'
        if  ' ' != qa_pair[-1]:
            qa_pair += ' '
        pattern = r'#\d'
        matches = re.findall(pattern, qa_pair)
        for match in matches:
            j = int(match[1]) # step number
            if j < 1: continue
            qa_pair = qa_pair.replace(match, qa_pairs[j-1].split('? ')[-1])
        qa_pairs.append(qa_pair + qa['answer'])

    return qa_pairs

### For PRM800K ###
def prompt_generation_prm800k(d, include_answer=False, evaluate=False):
    prompt = (
        'Please answer the following math question step-by-step. The output should follow the rules:\n'
        '1. Each step should be in a new line and start with "Step <number>: " indicating the step index.\n'
        '2. If the final answer is deduced, please follow the format, "# Answer\\n\\n<answer>", to output '
        'your final answer individually in the last line.\n'
        '3. If the number of steps is larger than 50 or reasoning steps ends with no proper answer, please only output '
        '"No Answer" in the last line.'
    )

    question = 'Question: ' + d['question']['problem']

    if evaluate:
        prompt =  question

    if include_answer:
        answer = answer_generation(d)
        prompt = '\n\n'.join([prompt, question, answer])
    else:
        prompt = '\n\n'.join([prompt, question])

    return prompt

def answer_generation_prm800k(d):
    answer = []
    for idx, step in enumerate(d['question']['pre_generated_steps']):
        step = f'Step {idx+1}: ' + step
        answer.append(step)
    answer = '\n'.join(answer)

    return answer

### For MATH-Shepherd ###
def prompt_generation_gsm8k(d, include_answer=False, evaluate=False):
    prompt = (
        'Please answer the following math question step-by-step. The output should follow the rules:\n'
        '1. Each step should be in a new line and start with "Step <number>: " indicating the step index.\n'
        '2. If the final answer is deduced, please follow the format, "# Answer\\n\\n<answer>", to output '
        'your final answer individually in the last line.\n'
        '3. If the number of steps is larger than 50 or reasoning steps ends with no proper answer, please only output '
        '"No Answer" in the last line.'
        )

    question = 'Question: ' + d['question']

    if evaluate:
        prompt = question

    if include_answer:
        answer = answer_generation(d)
        prompt = '\n\n'.join([prompt, question, answer])
    else:
        prompt = '\n\n'.join([prompt, question])

    return prompt

### For MATH500 ###
def prompt_generation_math500(d, include_answer=False, evaluate=False):
    prompt = (
        'Please answer the following math question step-by-step. The output should '
        'follow the rules:\n'
        '1. Each step should be in a new line and start with "Step <number>: " '
        'indicating the step index.\n'
        '2. If the final answer is deduced, please follow the format, '
        '"# Answer\\n\\n<answer>", to output your final answer individually in the '
        'last line.\n'
        '3. If the number of steps is larger than 50 or reasoning steps ends with '
        'no proper answer, please only output "No Answer" in the last line.'
        )

    question = 'Question: ' + d['problem']

    if evaluate:
        prompt = question

    if include_answer:
        answer = answer_generation(d)
        prompt = '\n\n'.join([prompt, question, answer])
    else:
        prompt = '\n\n'.join([prompt, question])

    return prompt

def answer_generation_math500(d):
    answer = []

    for idx, step in enumerate(d['solution'].split('  ')):
        steps = f'Step {idx+1}: ' + step 
        answer.append(steps)
    answer.append(f"# Answer\n\n{d['answer']}")
    answer = '\n'.join(answer)
    return answer

### For GSM8k ###
def answer_generation_gsm8k(d):
    answer = []

    for idx, step in enumerate(d['cot']):
        steps = f'Step {idx+1}: ' + step 
        answer.append(steps)
    answer.append(f"Step {idx+1}: # Answer\n\n{d['answer']}")
    answer = '\n'.join(answer)
    return answer

### synthetic reasoning steps ###
def synthetic_prompt_generation(d):
    ''' Six types of hallucination
    Factual Inconsistency
    Factual Fabrication
    Instruction inconsistency
    Context inconsistency
    Logical inconsistency
    Caculatio Error
    '''
    synthetic_prompt = (
        'The above are step-wise answers to the question. The answer may not be complete but all steps are correct. '
        'Please modify the last step to include the consistency hallucination by following the instructions:\n'
        '1. If the step refers to results from its previous steps, it should not directly refer to them.\n'
        '2. You can create similar/relevant results as referred the results.\n'
        '3. You can also modify the referred results to make them look different.\n\n'
        'Please only output the last step. The step starts with "Step <number>: ", where the number is its original step index.'
    )

    question = 'Question: ' + d['question']['problem']
    answer = answer_generation(d)
    prompt = '\n\n'.join([question, answer, synthetic_prompt])

    return prompt


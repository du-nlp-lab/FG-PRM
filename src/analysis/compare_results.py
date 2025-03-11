import json

from tqdm import tqdm

from openai_api import call
from prompt import prompt_generation, answer_generation

golden_input_file_name = '../../data/musique_sample_50.json'
predict_input_file_name = '../../data/musique_sample_generations_gpt-3.5.json'
compare_output_file_name = '../../data/musique_sample_compare_completeness_gpt-3.5.json'

with open(golden_input_file_name, 'r') as f:
    golden_data = json.load(f)
    golden_data = golden_data[:50]

with open(predict_input_file_name, 'r') as f:
    predict_data = json.load(f)

compare = {}

for d in tqdm(golden_data):
    idx = d['id']
    if idx not in compare:
        compare[idx] = []

    context = prompt_generation(d, evaluate=True)
    golden_answer = 'Golden ' + answer_generation(d)

    predict = predict_data[idx]
    for p in predict:
        predict_answer = 'Generated ' + p

        # Correctness
        prompt = ''
        # Faithfulness
        prompt = 'According to the above documents and the question, please evaluate whether the all steps in generated reasoning processes and answers are correct and faithful to the context. 
        Please only output Yes or No in the first line. If it is no, please output which step in the generated reasoning processes is incorrect in the reasoning process and summarize why it is incorrect.'
        # Completeness
        prompt = 'According to the above documents and the question, please evaluate whether the generated reasoning processes are complete. Please only output Yes or No in the first line.  If it is no, please output which step in generated reasoning processes is not complete and summarize why it is incomplete. Additionally, in the last line, please only output an integer indicating the step number (start from 1) only when the first line is No.'
        # Consistency
        prompt = 'According to the above documents and the question, please evaluate all steps in the generated reasoning processes. For each step, please determine whether it is logically consistency to all previous steps. Logically consistency indicates the current step has no logical contradiction to any previous step. If a step satisfies the creterion, please output "Yes". Otherwise, please output "No", follow by the reasoning, and indicate which previous step has logical contradiction with the current step at last by following the format: Step <number>.'

        prompt = '\n\n'.join([context, golden_answer, predict_answer, prompt])

        output = call(prompt)

        compare[idx].append(output)

with open(compare_output_file_name, 'w') as f:
    json.dump(compare, f, indent=2)


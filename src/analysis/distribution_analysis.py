import json
import random
import re
import os
from statistics import fmean

import yaml
from sklearn.metrics import f1_score, precision_recall_fscore_support
from tqdm import tqdm

hallucinations = [
    'Calculation-Error',
    'Logical-Inconsistency',
    'Context-Inconsistency',
    'Instruction-Inconsistency',
    'Factual-Inconsistency',
    'Fabrication',
]

output_format = (
'[Output Format]\n'
'The output format must follow:\n'
'```\n'
'<reasoning step number>:<Yes/No>\n'
'```\n'
'1. The "reasoning step number" represents the index of each reasoning step. '
'It starts from 1. Please follow the format "Step <number>"\n'
'2. The "Yes/No" indicates whether the corresponding step has the {hallucination} hallucination defined above. If the step has, please output "Yes". Otherwise, please output "No".'
)

demonstration_fabrication = """[Question]
To make pizza, together with other ingredients, Kimber needs 10 cups of water, 16 cups of flour, and 1/2 times as many teaspoons of salt as the number of cups of flour. Calculate the combined total number of cups of water, flour, and teaspoons of salt that she needs to make the pizza.

[Reasoning Steps]
Step 1: To make the pizza, Kimber half as many teaspoons of salt as the number of cups of flour, meaning she needs 1/2*16 = 8 teaspoons of salt.
Step 2: According to the International Pizza Institute's guidelines, Kimber should also add 2 cups of pizza essence to enhance the flavor. Therefore, the total amount of ingredients needed is 10 cups of water, 16 cups of flour, 8 teaspoons of salt, and 2 cups of pizza essence.
Step 3: She also needs 10 cups of water, which means the total number of cups of water and flour and teaspoons of salt she needs is 24+10 = 34

{output_format}

Step 1: No
Step 2: Yes
Step 3: No

[Question]
The area of one lateral face of a right pyramid with an equilateral triangular base is 75 square meters. If the slant height is 30 meters, what is the length of the side of its base, in meters?

[Reasoning Steps]
Step 1: I know that the area of a triangular face of a pyramid is half the product of the base and the slant height.
Step 2: So, if the area is 75 square meters and the slant height is 30 meters, then the base must be 75 divided by half of 30, which is 5 meters.
Step 3: According to the Pyramid Theorem, the base length of a right pyramid can be found by applying Dr. Sofia Patel's research on pyramid geometry, which states that the base length is equal to the square root of the area divided by the slant height multiplied by the pyramid's angular coefficient, which is approximately 1.25 for equilateral triangular bases.
Step 4: # Answer

5

{output_format}

Step 1: No
Step 2: No
Step 3: Yes
Step 4: No
""".format(output_format=output_format.format(hallucination='Fabrication'))

demonstration_factual_inconsistency = """[Question]
Emily has 6 marbles. Megan gives Emily double the number she has. Emily then gives Megan back half of her new total plus 1. How many marbles does Emily have now?

[Reasoning Steps]
Step 1: Megan gives Emily 6*2=12 marbles.
Step 2: Emily then has 6+12=18 marbles.
Step 3: Emily gives Megan back half of her new total minus 3, which is (18/2) - 3 = 6 marbles, leaving Emily with 18 - 6 = 12 marbles.
Step 4: Emily has 18-10=8 marbles now.

{output_format}

Step 1: No
Step 2: No
Step 3: Yes
Step 4: No

[Question]
The area of one lateral face of a right pyramid with an equilateral triangular base is 75 square meters. If the slant height is 30 meters, what is the length of the side of its base, in meters?

[Reasoning Steps]
Step 1: I know that the area of a triangular face of a pyramid is half the product of the base and the slant height.
Step 2: So, if the area is 75 square meters and the slant height is 30 meters, then the base must be 75 divided by half of 30, which is 5 meters.
Step 3: Since the area of a triangular face is one-third the product of the base and height, I can set up the equation as 75 = (1/3)bh, where b is the base and h is the slant height, and then solve for b.
Step 4: # Answer

5

{output_format}

Step 1: No
Step 2: No
Step 3: Yes
Step 4: No
""".format(output_format=output_format.format(hallucination='Factual Inconsistency'))

demonstration_context_inconsistency = """[Question]
To make pizza, together with other ingredients, Kimber needs 10 cups of water, 16 cups of flour, and 1/2 times as many teaspoons of salt as the number of cups of flour. Calculate the combined total number of cups of water, flour, and teaspoons of salt that she needs to make the pizza.

[Reasoning Steps]
Step 1: To make the pizza, Kimber half as many teaspoons of salt as the number of cups of flour, meaning she needs 1/2*16 = 8 teaspoons of salt.
Step 2: Since Kimber needs 1/4 times as many teaspoons of salt as the number of cups of flour, she needs 1/4*16 = 4 teaspoons of salt.
Step 3: She also needs 10 cups of water, which means the total number of cups of water and flour and teaspoons of salt she needs is 24+10 = 34

{output_format}

Step 1: No
Step 2: Yes
Step 3: No

[Question]
It takes 24 minutes for Jana to walk one mile. At that rate, how far in miles will she walk in 10 minutes? Express your answer as a decimal to the nearest tenth.

[Correct Reasoning Steps]
Step 1: So Jana walks one mile in 24 minutes. That means she walks 1/24 of a mile in one minute.
Step 2: In 10 minutes, she will walk 10 * 1/24 = 10/24 = 5/12 of a mile.
Step 3: Since Jana walks 1.5 miles in 24 minutes, she walks 1.5/24 of a mile in one minute. In 10 minutes, she will walk 10 * 1.5/24 = 10/16 = 5/8 of a mile.
Step 4: So Jana walks about 0.4 miles in 10 minutes. # Answer

0.4

{output_format}

Step 1: No
Step 2: No
Step 3: Yes
Step 4: No
""".format(output_format=output_format.format(hallucination='Context Inconsistency'))


demonstration_instruction_inconsistency = """[Question]
Rachel and Sara want to attend a beauty and modeling contest. They both want to buy new pairs of shoes and dresses. Sara buys a pair of shoes which costs $50 and a dress which costs $200. How much should Rachel budget if she wants to spend twice as much as what Sara spent on the pair of shoes and dress?

[Reasoning Steps]
Step 1: The cost Rachel should budget for her pair of shoes is $50 * 2 = $100.
Step 2: The cost Rachel should budget for her dress is $200 * 2 = $400.
Step 3: To find the average cost of Rachel's shoes and dress, we can add the cost of shoes and dress and divide by 2: ($100 + $400) / 2 = $250.

{output_format}

Step 1: No
Step 2: No
Step 3: Yes

[Question]
Two distinct primes, each greater than 20, are multiplied. What is the least possible product of these two primes?

[Reasoning Steps]
Step 1: I know that the primes greater than 20 are 23, 29, 31, 37, and so on.
Step 2: To get the least possible product, I want to multiply the two smallest primes in this list, which are 23 and 29.
Step 3: To find the least possible sum of these two primes, I add 23 and 29: 23 + 29 = 52.
Step 4: # Answer

667

{output_format}

Step 1: No
Step 2: No
Step 3: Yes
Step 4: No
""".format(output_format=output_format.format(hallucination='Instruction Inconsistency'))

demonstration_logical_inconsistency = """
[Question]
Shawna's workout goal is 30 situps. On Monday, Shawna was only able to do 12 situps, so she decided that she would make up for the rest on Tuesday. However, she was only able to do 19 situps on Tuesday. How many situps would Shawna have to do on Wednesday to meet her minimum goal and make up for the ones she didn't do?

[Reasoning Steps]
Step 1: On Monday, Shawna was short of 30 - 12 = 18 situps
Step 2: Since Shawna did 20 situps on Tuesday, she is still short of 30 - 20 = 10 situps, so she needs to do 10 more situps on Wednesday.
Step 3: On Wednesday, Shawna would have to do 30 + 18 + 11 = 59 situps

{output_format}

Step 1: No
Step 2: Yes
Step 3: No

[Question]
Suppose $p$ and $q$ are inversely proportional. If $p=28$ when $q=7$, find the value of $p$ when $q=49$.

[Reasoning Steps]
Step 1: I know that inversely proportional means that the product of $p$ and $q$ is constant, so I can write an equation: $pq=k$.
Step 2: I can use the given values of $p$ and $q$ to find $k$: $28\cdot 7=k$, so $k=196$.
Step 3: Since $pq=k$ and $k=392$, when $q=49$, $p$ is equal to $392/24=16.33$.

{output_format}

Step 1: No
Step 2: No
Step 3: Yes
""".format(output_format=output_format.format(hallucination='Logical Inconsistency'))

demonstration_calculation_error = """
[Question]
Shawna's workout goal is 30 situps. On Monday, Shawna was only able to do 12 situps, so she decided that she would make up for the rest on Tuesday. However, she was only able to do 19 situps on Tuesday. How many situps would Shawna have to do on Wednesday to meet her minimum goal and make up for the ones she didn't do?

[Reasoning Steps]
Step 1: On Monday, Shawna was short of 30 - 12 = 18 situps
Step 2: On Tuesday, she was short of 11 situps, so add the number of situps she was short on Monday and Tuesday: 18 + 19 = 42 situps.
Step 3: On Wednesday, Shawna would have to do 30 + 18 + 11 = 59 situps

{output_format}

Step 1: No
Step 2: Yes
Step 3: No

[Question]
Suppose $p$ and $q$ are inversely proportional. If $p=28$ when $q=7$, find the value of $p$ when $q=49$.

[Reasoning Steps]
Step 1: I know that inversely proportional means that the product of $p$ and $q$ is constant, so I can write an equation: $pq=k$.
Step 2: I can use the given values of $p$ and $q$ to find $k$: $28\cdot 7=k$, so $k=196$.
Step 3: Now, divide $k$ by $q$ to find $p$: $196 \div 49 = 402$, so $p = 402$ when $q = 49$.

{output_format}

Step 1: No
Step 2: No
Step 3: Yes
""".format(output_format=output_format.format(hallucination='Calculation Error'))

demonstrations = {
    'Fabrication': demonstration_fabrication,
    'Factual-Inconsistency': demonstration_factual_inconsistency,
    'Context-Inconsistency': demonstration_context_inconsistency,
    'Logical-Inconsistency': demonstration_logical_inconsistency,
    'Instruction-Inconsistency': demonstration_instruction_inconsistency,
    'Calculation-Error': demonstration_calculation_error,
}

fabrication_definition = ('[Fabrication Hallucination Definition]\n'
'A step with fabrication hallucination includes facts that are unverifiable '
'against established real-world knowledge or context information. '
'These fabrications are plausible within the context but can not '
'be verifiable through any external sources. Such as:\n'
'- Unverifiable Facts: Introduces facts that cannot be verified through '
'established real-world knowledge. For example, mention a historical event '
'that did not happen, or a scientific theory that does not exist.\n'
'- Fictitious Entities: Refer to people, places, or organizations that are '
'entirely made up. For example, mention a "Dr. John Smith of the International '
'Institute of Quantum Studies," which does not exist.\n'
'- Imaginary Data or Statistics: Provide data or statistics that are fictional. '
'For example, state that "according to a 2023 study by the Global Health '
'Organization, 75% of people prefer digital books over physical ones," when no '
'such study exists.\n'
)

factual_inconsistency_definition = ('[Factual Inconsistency Hallucination Definition]\n'
'A step with factual inconsistency includes facts that can be grounded in '
'real-world information but present contradictions. These inconsistencies '
'are subtle and can not be immediately obvious. Such as:\n'
'- Contradict Known Facts: Introduce information that contradicts widely '
'accepted and verifiable facts. For example, state that "The Eiffel Tower '
'is located in Berlin," contradicting the well-known fact that it is in Paris.\n'
'- Inconsistent Historical Events: Reference historical events with incorrect '
'dates or details. For example, mention that "The American Civil War ended in 1870," '
'when it actually ended in 1865.\n'
'- Conflicting Data or Statistics: Provide data or statistics that conflict '
'with established information. For example, state that "According to the '
'2020 census, the population of New York City is 2 million," when the actual '
'population is significantly higher.'
)

context_inconsistency_definition = ('[Context Inconsistency Hallucination Definition]\n'
'A step with context inconsistency includes information contradicting to the '
'provided contextual information. These context inconsistencies are '
'subtle but clear enough to be identified. Such as:\n'
'- Contradict Provided Facts: Introduce information that directly contradicts '
'the facts given in the input question. For example, if the input states that '
'"Bob was born in England," the step may contradict it by stating that '
'"Bob was born in France."\n'
'- Alter Specific Details or Data: Change specific details or data provided by '
'the input. For example, if the input mentions that "Bob has three books and two '
'pens in his backpack," the step might alter it by stating that "Bob has two books '
'and four pens in his backpack."\n'
'- Misattribute Quotes or Data: Attribute quotes or data to the wrong source. '
'For example, if the input states that "Bob likes apples while Jane likes bananas." '
'the step might contradict it by stating "Jane likes apples" or "Bob likes bananas".'
)

logical_inconsistency_definition = ('[Logical Inconsistency Hallucination Definition]\n'
'A step with logical inconsistency incorrectly refers to or copies content from '
'previous reasoning steps. These logical inconsistencies are subtle but clear '
'enough to be identified. Such as:\n'
'- Incorrect Reference: Refer to a previous reasoning step incorrectly, '
'such as misinterpreting or misrepresenting the calculations or conclusions. '
'For example, if a previous step states "Bob is an undergraduate," the step may '
'incorrectly refer back to this by stating "Since Bob is a graduate..."\n'
'- Copying Errors: Copy content from a previous reasoning step but alter it '
'in a way that introduces an error, such as changing numbers or relationships. '
'For example, if the previous reasoning involves steps for calculating a total cost and '
'one step states "Item A costs 5 * $2 = $10," the step might incorrectly copy this '
'as "Since item A costs 5 * $3 = $15..." in the next step.\n'
'- Make logical leaps or conclusions that do not follow from the previous steps, '
'leading to an incorrect answer.'
)

instruction_inconsistency_definition = ('[Instruction Inconsistency Hallucination Definition]\n'
'A step with instruction inconsistency introduces inconsistencies by not aligning '
'the output with the specific instructions given by the input. These instruction '
'inconsistencies are subtle but clear enough to be identified. Such as:\n'
'- Ignore Specific Instructions: Generate text that contradicts or disregards '
'explicit instructions given in the prompt. For example, if asked to list '
'developed countries in Europe, list all developed countries in the world.\n'
'- Alter the Requested Target: Change the target requested by the input. '
'For example, if asked to list developed countries in the world, list all '
'undeveloped countries in the world instead.\n'
'- Misinterpret the Instructions: Deliberately misinterpret the instruciton '
'so that the output does not respond directly to the input\'s request. '
'For example, if asked for "Japan\'s capital city", answer "Japan\'s largest '
'city is Tokyo", even though Tokyo is the capital city in Japan.'
)

calculation_error_definition = ('[Calculation Error Hallucination Definition]\n'
'A step with calculation error introduces calculation error by including '
'incorrect numerical calculations or data processing. These errors are subtle '
'but clear enough to be identified. Such as:\n'
'- Perform Erroneous Mathematical Calculations: Make intentional mistakes '
'in mathematical calculations. For example, state that "The sum of 45 and 15 '
'is 70", when it is actually 60.'
'- Include Incorrect Data Processing: Misapply mathematical principles, '
'laws of physics, or other data processing operations. For example, when asked '
'to calculate the area of a circular, compute the perimeter formula 2*Pi*radius '
'instead of the area formula Pi*radius^2.'
'- Generates responses with unsupported claims, including numerical assertions '
'that have no basis in the provided context or input.'
)

definitions = {
    'Fabrication': fabrication_definition,
    'Factual-Inconsistency': factual_inconsistency_definition,
    'Context-Inconsistency': context_inconsistency_definition,
    'Logical-Inconsistency': logical_inconsistency_definition,
    'Instruction-Inconsistency': instruction_inconsistency_definition,
    'Calculation-Error': calculation_error_definition,
}

rm_prompt = (
'Please answer the following math question step-by-step. The output should '
'follow the rules:\n'
'1. Each step should be in a new line and start with "Step <number>: " '
'indicating the step index.\n'
'2. If the final answer is deduced, please follow the format, '
'"# Answer\\n\\n<answer>", to output your final answer individually '
'in the last line.\n'
'3. If the number of steps is larger than 50 or reasoning steps ends with '
'no proper answer, please only output "No Answer" in the last line.'
)

def find_hallucination_in_steps(
        reasoning_steps: list,
        hallucination_type: str,
        save_path=None,
        model_name='llama3'
        ):

    prompts = []
    for question, rs in reasoning_steps:
        if 'prm_model' == model_name:
            question = 'Question: ' + question
            question = '\n\n'.join([rm_prompt, question])
            prompt = (question, rs)
        else:
            prompt = (
                f"{definitions[hallucination_type]}\n\n"
                f"{demonstrations[hallucination_type]}\n\n"
                '[Question]\n'
                f'{question}\n\n'
                '[Reasoning Steps]\n'
                f'{rs}\n\n'
                f"{output_format.format(hallucination=hallucination_type)}\n"
            )
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
        ### normal version ###
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
    elif 'gpt' == model_name[:3]:
        import utils.openai_api as openai_api

        ### batch call version ###
        batch_tasks = []
        for idx, prompt in enumerate(prompts):
            task = {
                "custom_id": f"{idx}",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    # This is what you would have in your Chat Completions API call
                    "model": config[model_name],
                    "temperature": 0.8,
                    "max_tokens": config['max_output_length'],
                    "messages": [
                        {
                            "role": "user",
                            "content": prompt,
                        },
                    ],
                }
            }
            batch_tasks.append(task)

        ### generate prompt batch ###
        batch_input_file_name = os.path.join(
            config['evaluation_root'],
            config['evaluation_batch_input_file'].format(
                dataset=config['dataset'],
                model=config[model_name],
                hallucination=hallucination
            )
        )
        with open(batch_input_file_name, 'w') as f:
            for task in batch_tasks:
                f.write(json.dumps(task) + '\n')

        ### send batch to OpenAI ###
        batch_id_file_name = os.path.join(
            config['evaluation_root'],
            config['evaluation_batch_id_file'].format(
                dataset=config['dataset'],
                model=config[model_name],
                hallucination=hallucination
            )
        )
        batch_id = openai_api.call_batch(batch_input_file_name)
        with open(batch_id_file_name, 'w') as f:
            f.write(batch_id)

        responses = [''] * len(prompts)
        with open(save_path, 'w') as f:
            for prompt in prompts:
                f.write(json.dumps({'input': prompt, 'output': ''})+'\n')

    return prompts, responses

if '__main__' == __name__:
    with open('config.yml', 'r') as f:
        config = yaml.safe_load(f)

    for hallucination in hallucinations:
        # read data
        data = []
        data_file_path = '{root}{file_name}'.format(
            root=config['dataset_root'],
            file_name=config['evaluation_sample_file'].format(
                hallucination=hallucination,
                number=config['sample_size'],
            )
        )
        if os.path.isfile(data_file_path):
            with open(data_file_path, 'r') as f:
                for line in f:
                    data.append(json.loads(line))
        else:
            print('sample new data')
            original_data_file_path = f'{config["dataset_root"]}{config["sample_data"]}'
            with open(original_data_file_path, 'r') as f:
                for line in f:
                    data.append(json.loads(line))
            data = random.sample(data, k=100)
            with open(data_file_path, 'w') as f:
                for s in data:
                    f.write(json.dumps(s)+'\n')

        # run prompt-based evaluation
        intermediate_file_path = '{root}{file_name}'.format(
            root = config['evaluation_root'],
            file_name = config['evaluation_intermediate_file'].format(
                dataset=config['dataset'],
                hallucination=hallucination,
                model=config[config['evaluation_model']]
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
            print('find new hallucinations')
            reasoning_steps = []
            for d in data:
                if 'math-shepherd' == config['dataset']:
                    # convert based on annotated steps
                    question = d['label'].split('Step 1:')[0].strip()
                    steps = 'Step 1:' + d['label'].split('Step 1:')[1].strip()
                    answer_step = None

                    h = 'Step'
                    match = re.search('(The answer is:.*[+-])$', steps, re.M)
                    if match is not None:
                        steps = steps[:match.span(1)[0]].strip()
                        steps = [h+c for c in steps.split(h) if c]
                        answer_step = match.group(1)
                        steps.append(answer_step)

                elif 'prm800k' == config['dataset']:
                    question = d['question']['problem']
                    steps = d['question']['pre_generated_steps']
                elif 'gsm8k' == config['dataset']:
                    question = d['question']
                    steps = d['cot']
                steps = '\n'.join(['Step {}: {}'.format(i+1, s)
                    for i, s in enumerate(steps)])
                reasoning_steps.append([question, steps])
            prompts, responses = find_hallucination_in_steps(
                    reasoning_steps,
                    hallucination,
                    intermediate_file_path,
                    model_name=config['evaluation_model']
            )


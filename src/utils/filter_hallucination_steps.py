import re

import utils.openai_api as openai_api
import utils.llama3_api as llama3_api

logical_error_output_instruction = (
    'The above are step-wise reasoning steps to answer the question. '
    'Please help me determine whether the last reasoning step involves reasoning process. '
    'If the last reasoning step only refers contents in the question or previous reasoning steps and does not derive new content, the step does not involves reasoning proccess.\n'
    'In the output, there should be explanation whether the last reasoning step has reasoning process first. '
    'Then, in the new line, please only output "Yes" if the last reasoning step has calculation process. Otherwise, please only output "No".'
)

calculation_error_output_instruction = (
    'The above are step-wise reasoning steps to answer the question. '
    'Please help me determine whether the last reasoning step involves calculation processes, '
    'including mathematical calculations or formulas:\n'
    '- Mathematical Calculations: the step should have at least one calculation process. '
    'The calculation processes should include numbers (3, 5, 10 etc.) or mathematical symbols (sin, cos, x, y, PI, etc.), '
    'and they should be like "The sum of 45 and 15 is 60", "30*4+5=125", "sin(x)+cos(x)", etc.\n'
    '- Formulas: the step should include mathematical principles, laws of physics, or other data processing operations. '
    'Formulas may be in latex format. They can be simply stated in the step and do not have equal symbols. '
    'For example, formula can be "Pi*radius^2", "2*Pi*radius", "\\[sin(x)+cos(x)\\]", etc.\n'
    'In the output, there should be explanation whether the last reasoning step has calculation process first. '
    'Then, in the new line, please only output "Yes" if the last reasoning step has calculation process. Otherwise, please only output "No".'
)

factual_inconsistency_output_instruction = (
    'The above are step-wise reasoning steps to answer the question. '
    'Please help me determine whether the last reasoning step refers factual information not mentioned before the step. '
    'All factual information should be gounded in real-world information, including:\n'
    '- Known Geographic Facts: the step should include widely accepted and verifiable facts in its original format or name. '
    'For example, state the fact that "The Eiffel Tower is located in Paris.", '
    '"Mount Everest, the tallest mountain in the world, is located in the Himalayas.", etc.\n'
    '- Historical Events: the step should refer historical events with correct dates or details. '
    'For example, mention that "The American Civil War ended in 1865."\n'
    '- Factual Scientific Data or Statistics: the step should include correct real-world data or statistics. '
    'But, basic calculation process should not be counted as factual information.'
    'For example, a step can state that "According to the 2020 census, the population on earth is over 7.5 billion.", '
    '"There is 7 days a week.", "The pythagorean theorem is a^2+b^2=c^2.", etc.\n'
    'In the output, there should be explanation whether the last reasoning step has factual information and output the facutal information first. '
    'Then, in the new line, please only output "Yes" if the last reasoning step has factual information. Otherwise, please only output "No".'
)

context_inconsistency_output_instruction = (
    'The above are step-wise reasoning steps to answer the question. '
    'Please help me determine whether the last reasoning step refers question information. '
    'Referred content in the last reasoning step should be the same as it mentioned in the question. '
    'Contents indirectly related to the referred content, such as derived or concluded by the referred contents, should not be counted as question information.\n'
    'In the output, there should be an explanation whether the last reasoning step refers question information, '
    'output the extact referred question information in both the last reasoning step and question first. '
    'Then, in the new line, please only output "Yes" if the last reasoning step refers question information. Otherwise, please only output "No".'
)

logical_inconsistency_output_instruction = (
    'The above are step-wise reasoning steps to answer the question. '
    'Please help me determine whether the last reasoning step refers information in previous reasoning steps but not in the question. '
    'Referred content in the last reasoning step should be the same as it mentioned in the previous reasoning steps but not in the question. '
    'Contents indirectly related to the referred content, such as derived or concluded by the referred contents, should not be counted as previous information.\n'
    'In the output, there should be an explanation whether the last reasoning step refers information in  previous reasoning steps but not in the question, '
    'output the extact referred previous information in both the last reasoning step and previous step first. '
    'Then, in the new line, please only output "Yes" if the last reasoning step refers those information. Otherwise, please only output "No".'
)

output_instructions = {
    'Logical Error': logical_error_output_instruction,
    'Calculation Error': calculation_error_output_instruction,
    'Factual Inconsistency': factual_inconsistency_output_instruction,
    'Context Inconsistency': context_inconsistency_output_instruction,
    'Logical Inconsistency': logical_inconsistency_output_instruction,
}

def judge_last_steps_hallucination(
        correct_previous_reasoning_steps: list,
        hallucination_type: str,
        save_path=None
        ):
    if hallucination_type in ['Instruction Inconsistency', 'Fabrication']:
        # all golden steps must follow question and instructions, there is no need to judge them
        return [1] * len(correct_previous_reasoning_steps)

    output_instruction = output_instructions[hallucination_type]

    prompts = []
    for question, correct_reasoning_steps in correct_previous_reasoning_steps:
        prompt = (
            '[Question]\n'
            f'{question}\n\n'
            '[Reasoning Steps]\n'
            f'{correct_reasoning_steps}\n\n'
            '[Instruction]\n'
            f'{output_instruction}'
        )
        prompts.append(prompt)

    responses = llama3_api.batch_call(prompts, max_tokens=500, batch_size=16, save_path=save_path)

    flags = []
    for prompt, response in zip(prompts, responses):
        flag = 0
        match = re.search(r'(Yes|No)', response.split()[-1].strip())
        if match is not None: 
            content = match.group(1)
            flag = 1 if 'Yes'==content else 0
        match = re.search(r'(Yes|No)$', response.strip())
        if match is not None: 
            content = match.group(1)
            flag = 1 if 'Yes'==content else 0
        flags.append(flag)

    return flags


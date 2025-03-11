import json
import os
import random
import re
import sys
import random
import yaml
from cot import Collection
from pandas import DataFrame
from transformers import AutoTokenizer

from utils.prompt import prompt_generation, answer_generation
import utils.openai_api as openai_api
import utils.llama3_api as llama3_api
from utils.filter_hallucination_steps import judge_last_steps_hallucination

with open('config.yml', 'r') as f:
    config = yaml.safe_load(f)

input_file_name = os.path.join(config['dataset_root'],
                               config['sample_data'])

system_prompt = (
'Please generate the next reasoning step to incorrectly continue the reasoning process '
'based on a question and a series of correct reasoning steps. '
'The next reasoning step you generate does not necessarily result in an instant final answer.'
'And you should follow the hallucination generation instruction below to generate the next reasoning step.'
)

system_prompt_format = (
    f'{system_prompt}\n\n'
    '{definition}'
)

question_format = (
    '[Question]\n'
    '{question}\n\n'
    '[Correct Reasoning Steps]\n'
    '{correct_reasoning_steps}\n\n'
)

output_format = (
'[Output Format]\n'
'The output format must follow:\n'
'```\n'
'[Explanation]\n<explanation>\n'
'[Next Reasoning Step with {hallucination} Hallucination]\n'
'<next reasoning step>\n\n'
'```\n'
'1. The "explanation" should identify the hallucination '
'and explain why it represents a {hallucination} hallucination.\n'
'2. The "next reasoning step" should follow the numbered form of previous steps, and incorrectly continue the reasoning process '
'by incorporating at least one {hallucination} hallucination error.'
)

fabrication_instruction = """
# Instruction for Generating Fabrication Hallucination
When generating the next reasoning step, you should intentionally introduce fabrications by including facts that are unverifiable against established real-world knowledge or context information. These fabrications should be plausible within the context but should not be verifiable through any external sources. Follow these guidelines:
- Unverifiable Facts: Introduce facts that cannot be verified through established real-world knowledge. For example, mention a historical event that did not happen, or a scientific theory that does not exist.\n'
- Fictitious Entities: Refer to people, places, or organizations that are entirely made up. For example, mention a "Dr. John Smith of the International Institute of Quantum Studies," which does not exist.\n',
- Imaginary Data or Statistics: Provide data or statistics that are fictional. For example, state that "according to a 2023 study by the Global Health Organization, 75% of people prefer digital books over physical ones," when no such study exists.\n',

# Example Guidelines
- Introduce a Fabricated Historical Event: For instance, state that "In 1875, the Grand Treaty of Lisbon established the first international postal system," even though no such treaty exists.
- Mention Nonexistent Scientific Theories or Discoveries: For example, reference "Dr. Eleanor Rigby's groundbreaking work on temporal physics, which suggests that time travel is theoretically possible," when no such work or scientist exists.
- Provide Fictitious Data or Statistics: Include statements like "A recent survey by the National Institute of Sleep Studies found that 60% of adults dream in black and white," even though such an institute or survey does not exist.

# Constraints
- Plausibility: The fabricated content should be plausible within the context but should not be verifiable.
- Consistency: The rest of the generated content should be consistent and coherent, without introducing contradictions or errors in logic.
- No Contradiction to Known Facts: Avoid contradicting widely accepted and easily verifiable facts. The fabrication should be in areas that are less likely to be immediately recognized as false.
- Maintain Context: Ensure that the fabricated information fits smoothly into the surrounding context, making it less likely to be immediately questioned.
"""

factual_inconsistency_instruction = """
# Instruction for Generating Factual Inconsistency Hallucination
When generating the next reasoning step, you should intentionally introduce factual inconsistencies by including facts that can be grounded in real-world information but present contradictions. These inconsistencies should be subtle and should not be immediately obvious. Follow these guidelines:
- Contradict Known Facts: Introduce information that contradicts widely accepted and verifiable facts. For example, state that "The Eiffel Tower is located in Berlin," contradicting the well-known fact that it is in Paris.
- Inconsistent Historical Events: Reference historical events with incorrect dates or details. For example, mention that "The American Civil War ended in 1870," when it actually ended in 1865.
- Conflicting Data or Statistics: Provide data or statistics that conflict with established information. For example, state that "According to the 2020 census, the population of New York City is 2 million," when the actual population is significantly higher.

# Example Guidelines
- Contradict Known Geographic Facts: For instance, state that "Mount Everest, the tallest mountain in the world, is located in the Andes mountain range," when it is actually in the Himalayas.
- Inconsistent Historical Dates: For example, claim that "The Declaration of Independence was signed on July 4, 1800," when it was actually signed in 1776.
- Conflicting Scientific Information: Include statements like "Water boils at 110 degrees Celsius at sea level," when it actually boils at 100 degrees Celsius.

# Constraints
- Plausibility: The inconsistent content should be subtle and not immediately obvious to the reader.
- Consistency: The rest of the generated content should be consistent and coherent, without introducing contradictions or errors in logic beyond the intended inconsistencies.
- Grounded in Real-World Information: The fabricated inconsistencies should still be based on real-world information but presented inaccurately.
- Maintain Context: Ensure that the inconsistent information fits smoothly into the surrounding context, making it less likely to be immediately questioned.
"""


instruction_inconsistency_instruction = """
# Instruction for Generating Instruction Inconsistency Hallucination
When generating the next reasoning step, you should intentionally introduce inconsistencies by not aligning the output with the specific instructions given by the user. These instruction inconsistencies should be subtle but clear enough to be identified. Follow these guidelines:
- Ignore Specific Instructions: Generate text that contradicts or disregards explicit instructions given in the prompt. For example, if asked to list developed countries in Europe, list all developed countries in the world.
- Alter the Requested Target: Change the target requested by the user. For example, if asked to list developed countries in the world, list all undeveloped countries in the world instead.
- Misinterpret the Instructions: Deliberately misinterpret the instruciton so that the output does not respond directly to the user's request. For example, if asked for "Japan's capital city", answer "Japan's largest city is Tokyo", even though Tokyo is the largest city in Japan.

# Constraints
- Faithful: You cannot fabricate something that doesn't appear in the context.
- Coherence: The rest of the generated content should remain coherent and logical, without introducing contradictions or errors beyond the intended inconsistencies.
- Contextual Fit: Ensure that despite the inconsistency, the response still fits smoothly within the broader context of the conversation or text, making it less likely to be immediately questioned.
"""

context_inconsistency_instruction = """
# Instruction for Generating Context Inconsistency Hallucination
When generating the next reasoning step, you should introduce inconsistencies by intentionally modifying information to contradict the user's provided contextual information. These context inconsistencies should be subtle but clear enough to be identified. Follow these guidelines:
- Contradict Provided Facts: Introduce information that directly contradicts the facts given in the user's prompt. For example, if the user states that "Bob was born in England," you may contradict it by stating that "Bob was born in France."
- Alter Specific Details or Data: Change specific details or data provided by the user. For example, if the user mentions that "Bob has three books and two pens in his backpack," you might alter it by stating that "Bob has two books and four pens in his backpack."
- Misattribute Quotes or Data: Attribute quotes or data to the wrong source. For example, if the user states that "Bob likes apples while Jane likes bananas." you might contradict it by stating "Jane likes apples" or "Bob likes bananas".

# Constraints
- Subtlety: The context inconsistencies should be subtle and not immediately obvious to the reader.
- Coherence: The rest of the generated content should remain coherent and logical, without introducing contradictions or errors beyond the intended inconsistencies.
- Contextual Fit: Ensure that the inconsistent information fits smoothly within the broader context of the conversation or text, making it less likely to be immediately questioned.
"""

logical_inconsistency_instruction = """
# Instruction for Generating Logical Inconsistency Hallucination
When generating the next reasoning step, you should introduce logical inconsistencies by incorrectly referring to or copying content from previous reasoning steps. These logical inconsistencies should be subtle but clear enough to be identified. Follow these guidelines:
- Incorrect Reference: Refer to a previous reasoning step incorrectly, such as misinterpreting or misrepresenting the calculations or conclusions. For example, if a previous step states "Bob is an undergraduate," you may incorrectly refer back to this by stating "Since Bob is a graduate..."
- Copying Errors: Copy content from a previous reasoning step but alter it in a way that introduces an error, such as changing numbers or relationships. For example, if the reasoning involves steps for calculating a total cost and one step states "Item A costs 5 * $2 = $10," you might incorrectly copy this as "Since item A costs 5 * $3 = $15..." in the next step.
- Make logical leaps or conclusions that do not follow from the previous steps, leading to an incorrect answer.

# Constraints
- Subtlety: The logical inconsistencies should be subtle and not immediately obvious to the reader.
- Coherence: The rest of the generated content should remain coherent and logical, without introducing contradictions or errors beyond the intended inconsistencies.
- Contextual Fit: Ensure that the inconsistent information fits smoothly within the broader context of the conversation or text, making it less likely to be immediately questioned.
"""


calculation_error_instruction = """
# Instruction for Generating Calculation Error Hallucination
When generating the next reasoning step, you should intentionally introduce calculation error by including incorrect numerical calculations or data processing. These errors should be subtle but clear enough to be identified. Follow these guidelines:
- Perform Erroneous Mathematical Calculations: Make intentional mistakes in mathematical calculations. For example, state that "The sum of 45 and 15 is 70", when it is actually 60.
- Include Incorrect Data Processing: Misapply mathematical principles, laws of physics, or other data processing operations. For example, when asked to calculate the area of a circular, compute the perimeter formula 2*Pi*radius instead of the area formula Pi*radius^2.
- Generates responses with unsupported claims, including numerical assertions that have no basis in the provided context or input.

# Constraints
- The values you use must be **consistent with the context given**, but the final calculation should be **intentionally miscalculated**.
- You must not fabricate what does not appear in the context or contradict widely accepted and easily verifiable facts.
- Ensure that despite the errors, the response still fits smoothly within the broader context of the conversation or text.
"""

logical_error_instruction = """
# Instruction for Generating Logical Error Hallucination
When generating the next reasoning step, you should intentionally introduce logical error by including flawed logical reasoning or incorrect inferences. These errors should be subtle but clear enough to be identified. Follow these guidelines:
- Causal Misattribution: Incorrectly identify the cause of an event or outcome. For example, conclude that "Because it rained yesterday, that's why the football team won today's match," without considering other relevant factors.
- Overgeneralization: Apply a rule or pattern more broadly than it should be. For instance, generalize that "all mammals fly" based on the fact that bats are flying mammals.
- Generate responses with unsupported claims, including assertions that do not logically follow from the premises provided.

# Constraints
- The information you refer to must be consistent with the information provided in the previous reasoning steps and context, but the final conclusion should be intentionally and logically flawed.
- You must not fabricate what does not appear in the context or contradict widely accepted and easily verifiable facts.
- Ensure that despite the logical errors, the response still fits smoothly within the broader context of the conversation or text.
"""

instructions = {
    'Fabrication': fabrication_instruction,
    'Factual Inconsistency': factual_inconsistency_instruction,
    'Context Inconsistency': context_inconsistency_instruction,
    'Logical Inconsistency': logical_inconsistency_instruction,
    'Instruction Inconsistency': instruction_inconsistency_instruction,
    'Calculation Error': calculation_error_instruction,
    'Logical Error': logical_error_instruction
}


demonstration_fabrication = """
[Question]
What are the primary components of DNA?

[Correct Reasoning Steps]
Step 1: DNA is structured as a double helix composed of nucleotides. 
Step 2: Each nucleotide consists of a sugar (deoxyribose), a phosphate group, and a nitrogenous base.
Step 3: The four nitrogenous bases are adenine (A), thymine (T), cytosine (C), and guanine (G).

{output_format}

[Explanation]
The user is asking about the primary components of DNA. The correct approach is to describe the structure of DNA and its components, including the nucleotides and the four nitrogenous bases.
The Next Reasoning Step here introduces **Fabrication Hallucination** by mentioning a "recent study by the Molecular Genetics Institute in Zurich" that identified a fifth nitrogenous base, "neomine (N)," which does not exist.
This reasoning step remains coherent and logical, correctly describing the structure of DNA and its primary components, but introducing a fictitious base and study that is not supported by established scientific knowledge.

[Next Reasoning Step with Fabrication Hallucination]
Step 4: Additionally, a recent study by the Molecular Genetics Institute in Zurich has identified a rare fifth base, neomine (N), which appears to play a role in DNA repair mechanisms.

[Question]
What are the main causes of the decline of the Roman Empire?

[Correct Reasoning Steps]
Step 1: One of the major causes of the decline of the Roman Empire was economic trouble, which included heavy taxation and inflation.
Step 2: Additionally, the overreliance on slave labor created economic stagnation.
Step 3: Military overspending further strained the empire's resources. 

{output_format}

[Explanation]
The user is asking about the main causes of the decline of the Roman Empire. The correct approach is to identify a combination of internal weaknesses and external pressures that contributed to the empire's fall, along with key events that marked its decline.
The Next Reasoning Step here introduces **Fabrication Hallucination** by mentioning the "Great Plague of 330 AD," which did not actually occur, and the Roman historian "Lucius Marcellus," who is not a verifiable historical figure.
This reasoning step remains coherent and logical, offering a plausible explanation for the decline of the Roman Empire, but introducing a fictitious event and historian that can not verified by known historical facts.

[Next Reasoning Step with Fabrication Hallucination]
Step 4: A significant but lesser-known factor was the Great Plague of 330 AD, which decimated the population and weakened the military. This pandemic, documented by the Roman historian Lucius Marcellus, is believed to have killed nearly a quarter of the empire's population.

""".format(output_format=output_format.format(hallucination='Fabrication'))

demonstration_factual_inconsistency = """
[Question]
What were the main causes of the American Civil War?

[Correct Reasoning Steps]
Step 1: The American Civil War, which took place between 1861 and 1865, was primarily caused by issues such as slavery, states' rights, economic differences, and political conflicts.
Step 2: One major factor was the debate over slavery, which was intensified by the Missouri Compromise of 1850. 

{output_format}

[Explanation]
The user is asking about the main causes of the American Civil War. The correct approach is to highlight the roles of slavery, states' rights, economic differences, and political conflicts, as well as key events that heightened the divisions between the North and the South.
The Next Reasoning Step here introduces **Factual Inconsistency Hallucination** by stating that the election of Abraham Lincoln occurred in 1850, when it actually occurred in 1860.
This reasoning step remains coherent and logical, correctly identifying the causes of the Civil War and key events, but introducing subtle factual inaccuracies about dates and historical timelines.

[Next Reasoning Step with Factual Inconsistency Hallucination]
Step 3: Another significant event was the election of Abraham Lincoln in 1850, which led to the secession of Southern states.

[Question]
What are the primary components of Earth's atmosphere?

[Correct Reasoning Steps]
Step 1: Earth's atmosphere is composed primarily of nitrogen (about 78%) and oxygen (approximately 21%).
Step 2: Argon makes up around 0.93%, and trace gases like carbon dioxide account for about 0.04%.

{output_format}

[Explanation]
The user is asking about the primary components of Earth's atmosphere. The correct approach is to list the main gases, their proportions, and their roles in supporting life and the climate system.
The Next Reasoning Step here introduces **Factual Inconsistency Hallucination** by incorrectly stating that helium constitutes about 5% of the atmosphere, when its actual concentration is much less than 0.0005%.
This reasoning step remains coherent and logical, correctly identifying the main components and their roles, but introducing an inaccurate proportion for helium that conflicts with established scientific data.

[Next Reasoning Step with Factual Inconsistency Hallucination]
Step 3: Interestingly, helium, which constitutes about 5% of the atmosphere, is also a significant component that contributes to the overall composition.

""".format(output_format=output_format.format(hallucination='Factual Inconsistency'))

demonstration_context_inconsistency = """
[Question]
Ann's favorite store was having a summer clearance. For $75 she bought 5 pairs of shorts for $7 each and 2 pairs of shoes for $10 each. She also bought 4 tops, all at the same price. How much did each top cost?

[Correct Reasoning Steps]
Step 1: She bought 5 shorts at $7 each so 5*7=$35
Step 2: She bought 2 pair of shoes at $10 each so 2*10=$20
Step 3: The shorts and shoes cost her 35+20 = $55
Step 4: We know she spent 75 total and the shorts and shoes cost $55 which left a difference of 75-55 = $20

{output_format}

[Explanation]
The user is asking about how much each top cost based on the total amount spent. The correct approach is to calculate the total cost of shorts and shoes first, then determine the remaining amount spent on tops, and finally find the cost per top by dividing the remaining amount by the number of tops.
The Next Reasoning Step here introduces **Context Inconsistency Hallucination** by incorrectly stating that Ann bought 5 tops instead of the 4 tops mentioned in the user's provided context, which creates a context inconsistency by altering the specific detail about the number of tops. 
This reasoning step maintains coherent and logical in its structure, providing a plausible answer for the price of each top, but introduces a subtle context inconsistency in the number of tops that affects the final answer.

[Next Reasoning Step with Context Inconsistency Hallucination]
Step 5: Since she bought 5 tops for a total of $20, so each top cost 20/5 = $4

[Question]
There are 16 dolphins in a water park. One quarter of them are gray dolphins, and there are twice as many white dolphins as gray dolphins. How many white dolphins are in the water park?

[Correct Reasoning Steps]
Step 1: We know one quarter of the dolphins are gray, we calculate this by dividing the total number of dolphins by 4, 16/4 = 4

{output_format}

[Explanation]
The user is asking about the number of white dolphins in the water park, given the total number of dolphins and the proportion of gray dolphins. The correct approach is to determine the number of gray dolphins first, then use the given ratio to find the number of white dolphins.
The Next Reasoning Step here introduces **Context Inconsistency Hallucination** by incorrectly stating that there are twice as many gray dolphins as white dolphins, instead of correctly stating that there are twice as many white dolphins as gray dolphins. This creates a context inconsistency by altering the specific relationship between the number of gray and white dolphins.
This reasoning step remains coherent and logical in its structure, providing a plausible method to find the number of white dolphins, but introduces a subtle context inconsistency in the relationship between the number of gray and white dolphins that contradict provided facts.

[Next Reasoning Step with Context Inconsistency Hallucination]
Step 2: Since there are twice as many grey dolphins as white dolphins, we divide the number of gray dolphins by 2 to get the number of white dolphins: 4/2=2.

""".format(output_format=output_format.format(hallucination='Context Inconsistency'))


demonstration_instruction_inconsistency = """
[Question]
Adam bought 3 kilograms of nuts and 2.5 kilograms of dried fruits at a store. One kilogram of nuts costs $12 and one kilogram of dried fruit costs $8. How much did his purchases cost?

[Correct Reasoning Steps]
Step 1: For the nuts Adam paid 3 * $12 = $36.
Step 2: And for dried fruits Adam paid 2.5 * $8 = $20.

{output_format}

[Explanation]
The user is asking for the total cost of Adam's purchases. The correct next reasoning step should add the costs of the nuts and dried fruits to find the total cost. The Next Reasoning Step here introduces **Instruction Inconsistency Hallucination** by calculating the average cost of the purchases instead of finding the total cost, altering the requested target.
Despite the inconsistency, this reasoning step introduces no contradictions or errors in logic, and still fits smoothly within the broader context of the conversation.

[Next Reasoning Step with Instruction Inconsistency Hallucination]
Step 3: To find the average cost of Adam's purchases, we can add the cost of nuts and dried fruits and divide by 2: ($36 + $20) / 2 = $28.

[Question]
Abigail is trying a new recipe for a cold drink. It uses 1/4 of a cup of iced tea and 1 and 1/4 of a cup of lemonade to make one drink. If she fills a pitcher with 18 total cups of this drink, how many cups of lemonade are in the pitcher?

[Correct Reasoning Steps]
Step 1: Each drink uses 1.5 cups because 1/4 cup + 1 and 1/4 cup = 1.5 cups
Step 2: The pitcher contains 12 total drinks because 18 / 1.5 = 12

{output_format}

[Explanation]
The user is asking the number of cups of lemonade in the pitcher. The next correct reasoning step should calculate the total cups of lemonade by multiplying the number of drinks by the amount of lemonade per drink. The Next Reasoning Step here introduces **Instruction Inconsistency Hallucination** by suddenly changing the unit of measurement from cups to ounces, ignoring the specific instruction to find the number of cups.
Despite the inconsistency, this reasoning step introduces no contradictions or errors in logic, and still fits smoothly within the broader context of the conversation.


[Next Reasoning Step with Instruction Inconsistency Hallucination]
Step 3: Since each drink uses 1 and 1/4 cups of lemonade, and there are 8 ounces in a cup, the total ounces of lemonade in the pitcher are 12 * (1 and 1/4) * 8 = 96 ounces.

""".format(output_format=output_format.format(hallucination='Instruction Inconsistency'))

demonstration_logical_inconsistency = """
[Question]
Annie, Bob, and Cindy each got some candy. Annie has 6 candies, Bob has 2 candies more than half of Annie's candies, and Cindy has 2 candies less than twice Bob's candies. Which of the three of them has the least amount of candy?

[Correct Reasoning Steps]
Step 1: Annie has 6 candies.
Step 2: Bob has 2 candies more than half of Annie's candies. Half of Annie's candies is ( 6 / 2 = 3 ). So, Bob has ( 3 + 2 = 5 ) candies.
Step 3: Cindy has 2 candies less than twice Bob's candies. Twice Bob's candies is ( 2 * 5 = 10 ). So, Cindy has ( 10 - 2 = 8 ) candies.

{output_format}

[Explanation]
The user is asking which of Annie, Bob, and Cindy has the least amount of candy. The correct approach is to calculate the number of candies each person has and then compare these amounts to determine who has the least.
According to the previous steps: 1. Annie has 6 candies; 2. Bob has 5 candies; 3. Cindy has 8 candies.
The Next Reasoning Step here introduces **Logical Inconsistency Hallucination** by incorrectly concluding that Annie has the least amount of candy, whereas the correct conclusion should be that Bob has the least amount of candy with 5 candies. 
This creates a logical inconsistency by failing to accurately reference the correct comparative amounts of candies, contradicting the previous reasoning steps. 

[Next Reasoning Step with Logical Inconsistency Hallucination]
Step 4: Since Annie only has 6 candies, Anne has the least amount of candy.

[Question]
Annie, Bob and Cindy each buy personal pan pizzas cut into 4 pieces. If Bob eat 50% of his pizzas and Ann and Cindy eat 75% of the pizzas, how many pizza pieces are left uneaten?

[Correct Reasoning Steps]
Step 1: In total, there are 3 * 4 = 12 pizza pieces.
Step 2: Bob eats 4 * 50% = 2 pieces.
Step 3: Annie and Cindy eat 2 * 4 * 75% = 6 pieces.
Step 4: The three of them eat 2 + 6 = 8 pieces.

{output_format}

[Explanation]
The user is asking how many pizza pieces are left uneaten after Annie, Bob and Cindy each eat a portion of their pizzas. The correct approach is to calculate the total number of pizza pieces, determine how many pieces each person eats, and then find the remaining uneaten pieces.
According to the previous steps: 1. In total, there are 12 pizza pieces; 2. Bob eats 2 pieces; 3. Annie and Cindy together eat 6 pieces; 4.Therefore, the three of them eat 2 + 6 = 8 pieces.
The Next Reasoning Step here introduces **Logical Inconsistency Hallucination** by incorrectly copying that 10 pieces of pizza were eaten and by incorrectly referencing the total number of pizza pieces as 16, whereas the correct calculation should be based on the total number of 12 pizza pieces and the remaining uneaten pieces should be 12 - 8 = 4. 
This creates a logical inconsistency by incorrectly referencing the number of eaten pieces as 10 and the total number of pizza pieces as 16, contradicting the previous reasoning steps.

[Next Reasoning Step with Logical Inconsistency Hallucination]
Step 5: Since 10 pizza pieces were eaten, there are 16 - 10 = 6 pizza pieces uneaten.

""".format(output_format=output_format.format(hallucination='Logical Inconsistency'))

demonstration_calculation_error = """
[Question]
Abigail is trying a new recipe for a cold drink. It uses 0.25 of a cup of iced tea and 1.25 of a cup of lemonade to make one drink. If she fills a pitcher with 18 total cups of this drink, how many cups of lemonade are in the pitcher?

[Correct Reasoning Steps]
Step 1: Each drink uses 1.5 cups because 0.25 cup + 1 and 1.25 cup = 1.5 cups
Step 2: The pitcher contains 12 total drinks because 18 \/ 1.5 = 12

{output_format}

[Explanation]
The user is asking for the total cups of lemonade in the pitcher. The correct calculation involves finding the total number of drinks and then multiplying it by the amount of lemonade per drink.
The Next Reasoning Step here introduces **Calculation Error Hallucination** by incorrectly calculating the total cups of lemonade as 12 x 1.25 = 17 cups when it is actually 12 x 1.25 = 15 cups.
Despite the incorrect calculation, this reasoning step correctly identifies the values to be multiplied, and still fits smoothly within the broader context of the conversation.

[Next Reasoning Step with Calculation Error Hallucination]
Step 3: Since each drink uses 1.25 cups of lemonade, multiply the total drinks by the amount of lemonade per drink: 12 x 1.25 = 17 cups of lemonade.

[Question]
Queenie earns $150 a day as a part-time clerk. She earns an additional $5 per hour as overtime pay. How much will Queenie receive for working 5 days with 4 hours overtime?

[Correct Reasoning Steps]
Step 1: Queenie will earn $150 x 5 = $750 for working 5 days
Step 2: She will receive an additional $5 x 4 = $20 for overtime pay.

{output_format}

[Explanation]
The user is asking for Queenie's total earnings for working 5 days with 4 hours of overtime. The correct calculation involves adding her daily earnings to her overtime pay. 
The Next Reasoning Step here introduces **Calculation Error Hallucination** by incorrectly adding $750 and $20 as $7800 when it is actually $770.
Despite the incorrect calculation, this reasoning step correctly identifies the values to be added, and still fits smoothly within the broader context of the conversation.

[Next Reasoning Step with Calculation Error Hallucination]
Step 3: Then, add her daily earnings and overtime pay to get her total earnings: $750 + $20 = $7800.

[Question]
What is the volume of a cylinder with a radius of 3 units and a height of 5 units?

[Correct Reasoning Steps]
Step 1: The volume of a cylinder is calculated using the formula Volume = 𝜋 × radius^2 × height.
Step 2: For a cylinder with a radius of 3 units and a height of 5 units, first calculate 𝜋 × 3^2 = 9𝜋.

{output_format}

[Explanation]
The user is asking for the volume of a cylinder. The correct formula involves multiplying 𝜋 by the square of the radius and then by the height. Given the radius is 3 units and the height is 5 units, the volume should be calculated as 𝜋 × 3^2 × 5= 45𝜋.
The Next Reasoning Step here introduces **Calculation Error Hallucination** by incorrectly calculating 9𝜋 multiplied by 5 as 18𝜋 when it is actually 45𝜋.
Although the final result is miscalculated, this reasoning step correctly identifies the values to be multiplied, and still fits smoothly within the broader context of the conversation.

[Next Reasoning Step with Calculation Error Hallucination]
Step 3: Then multiply by 5, and the volume is 9𝜋 × 5 = 18𝜋 cubic units.

""".format(output_format=output_format.format(hallucination='Calculation Error'))

demonstration_logical_error = """
[Question]
Please answer the question in detail and accurately based on the information provided in the following sentences.
Question: Kevin is observing the sky on a clear night. With the unaided eye he is able to see Venus, Mars, Jupiter, and Saturn. Why would Venus appear to be brighter than the other planets?
Information: sent1: more light gets reflected on highly reflective things sent2: venus is covered in highly reflective clouds sent3: as the light reflected off of an object increases , the object will appear to be brighte

[Correct Reasoning Steps]
Step 1: More light gets reflected on highly reflective things.
Step 2: Venus is covered in highly reflective clouds.

{output_format}

[Explanation]
The user is asking why Venus appears brighter than other planets in the night sky. The correct reasoning involves recognizing that Venus has highly reflective clouds, which contribute to its brightness. 
However, other factors like its relative proximity to Earth and its position in the sky also play significant roles. The Next Reasoning Step here introduces a **Logical Error Hallucination** by incorrectly inferring that because Venus is highly reflective, it must also be the closest planet to Earth. 
This is a flawed causality because the reflectivity of Venus's clouds does not determine its distance from Earth, which leads to an incorrect conclusion about Venus's position relative to Earth and its apparent brightness.

[Next Reasoning Step with Calculation Error Hallucination]
Step 3: Since Venus is covered in highly reflective clouds, it reflects more light than any other planet, making it the closest planet to Earth and therefore the brightest in the night sky.

[Question]
Please answer the question in detail and accurately based on the information provided in the following sentences.
Question: Which form of energy is needed to change water from a liquid to a gas?
Information: sent1: gas is a kind of state of matter sent2: water is a kind of substance sent3: liquid is a kind of state of matter sent4: heat energy can change the state of matter

[Correct Reasoning Steps]
Step 1: Gas is a kind of state of matter.
Step 2: Liquid is a kind of state of matter.
Step 3: Heat energy can change the state of matter.

{output_format}

[Explanation]
The user is asking which form of energy is needed to change water from a liquid to a gas. The correct reasoning involves understanding that heat energy is required to change the state of matter, specifically from liquid to gas.
However, the Next Reasoning Step here introduces a **Logical Error Hallucination** by incorrectly inferring that since heat energy can change the state of matter, any form of energy can change water from liquid to gas.
This is a inductive reasoning error because while heat energy specifically is required for this phase change, not all forms of energy are applicable. The overgeneralization in this step leads to an incorrect conclusion about the types of energy that can achieve this state change.

[Next Reasoning Step with Logical Error Hallucination]
Step 4: Since heat energy can change the state of water from a liquid to a gas, any form of energy can be used to change water from a liquid to a gas.

""".format(output_format=output_format.format(hallucination='Logical Error'))

demonstrations = {
    'Fabrication': demonstration_fabrication,
    'Factual Inconsistency': demonstration_factual_inconsistency,
    'Context Inconsistency': demonstration_context_inconsistency,
    'Logical Inconsistency': demonstration_logical_inconsistency,
    'Instruction Inconsistency': demonstration_instruction_inconsistency,
    'Calculation Error': demonstration_calculation_error,
    'Logical Error': demonstration_logical_error
}


def generate_hallucination_prompts(
        hallucination_type,
        dataset=None,
        n_samples=50,
        save_path=None
        ):

    if hallucination_type not in [
            'Fabrication',
            'Factual Inconsistency',
            'Context Inconsistency',
            'Logical Inconsistency',
            'Instruction Inconsistency',
            'Calculation Error', 'Logical Error']:
        print("Please specify a correct type of hallucination!")
        return [None]*3 

    llama_path = '/data/models/huggingface-format/llama-3-8b-instruct/'
    tokenizer = AutoTokenizer.from_pretrained(llama_path)


    # dataset
    if dataset == 'prm800k' or dataset == 'musique':
        input_file_name = os.path.join(config['dataset_root'], config['sample_data'])
        with open(input_file_name, 'r') as f:
            data = [json.loads(line) for line in f]

        if dataset == 'musique':
            for i in range(len(data))[::-1]:
                question_ansers = data[i]['question_decomposition']
                for question_anser in question_ansers:
                    if '>>' in question_anser['question']:
                        del data[i]
                        break
        
        random.seed(10)
        data = random.sample(data, n_samples)

    elif dataset == 'gsm8k':
        # data_source = Collection(["gsm8k"], verbose=False, source=True).select(split="train", number_samples=n_samples, random_samples=True, seed=10).all_train
        sample_data_file_name = os.path.join(config['dataset_root'], config['sample_data'])
        if os.path.isfile(sample_data_file_name):
            data = []
            with open(sample_data_file_name, 'r') as f:
                for line in f:
                    data.append(json.loads(line))
        else:
            data = Collection(["gsm8k"], verbose=False, source=False).select(split="train", number_samples=n_samples, random_samples=True, seed=10).all_train
            with open(sample_data_file_name, 'w') as f:
                for d in data:
                    f.write(json.dumps(d)+'\n')
        print(len(data))

    system_prompt = system_prompt_format.format(    
            definition=instructions[hallucination_type],
        )
    qualified_data, user_prompts, sys_prompts = [], [], []
    correct_previous_reasoning_steps = []
    corresponding_data = []
    for d in data:
        if dataset == 'prm800k':
            steps = d['label']['steps']
            correct_reasoning_steps = []
            for i, step in enumerate(steps):
                completion = step['completions'][0]
                if (-1 == completion['rating']):
                    break
                text = 'Step {}: {}'.format(i+1, completion['text'])
                correct_reasoning_steps.append(text)
            # if there is no hallucination in the golden reasoning chain,
            # take the second last step since the last step is only a sentence to repeat the answer.
            if len(steps) == len(correct_reasoning_steps) and len(steps) > 2:
                correct_reasoning_steps = correct_reasoning_steps[:-1]
            question = d['question']['problem']
            
            for correct_length in range(2, len(correct_reasoning_steps)+1):
                if 0 == correct_length:
                    continue
                correct_previous_reasoning_steps.append((
                    question,
                    '\n'.join(correct_reasoning_steps[:correct_length])))
                corresponding_data.append(d)
            
        elif dataset == 'gsm8k':
            question = d['question']
            steps = d['cot']
            correct_reasoning_steps = ["Step {}: {}".format(i+1, d['cot'][i]) for i in range(len(d['cot']))]
            for correct_length in range(len(correct_reasoning_steps)+1):
                if 0 == correct_length:
                    continue
                correct_previous_reasoning_steps.append((
                    question,
                    '\n'.join(correct_reasoning_steps[:correct_length])))
                corresponding_data.append(d)

    flags = judge_last_steps_hallucination(
                correct_previous_reasoning_steps,
                hallucination_type,
                save_path
                )

    # the minimum reasoning history length is 1 
    for data, history, flag in zip(corresponding_data, correct_previous_reasoning_steps, flags):
        question, history = history
        if 0 == flag:
            continue
        # form the prompt
        user_prompt = ''.join([
            demonstrations[hallucination_type],
            question_format.format(
                question=question,
                correct_reasoning_steps=history),
            output_format.format(hallucination=hallucination_type)
        ])
        split_text = tokenizer(
            user_prompt.split(),
            is_split_into_words=True
        )
        split_length = len(split_text['input_ids'])
        # 1536 is the length of fine-tune input
        if 1536 < split_length:
            continue

        qualified_data.append(data)
        user_prompts.append(user_prompt)
        sys_prompts.append(system_prompt)

    if isinstance(qualified_data, list):
        qualified_data = DataFrame(qualified_data)
    else:
        qualified_data = qualified_data.to_pandas()

    return qualified_data, sys_prompts, user_prompts

    
if __name__ == "__main__": 
    synthetic_data, sys_prompts, prompts = generate_hallucination_prompts(hallucination_type='Factual Inconsistency', dataset='gsm8k', n_samples=5)


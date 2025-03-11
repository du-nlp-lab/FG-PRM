import json
from openai_api import call

with open('../../data/musique_sample_compare_gpt-3.5.json', 'r') as f:
    data = json.load(f)

reasons = []
for k, v in data.items():
    for r in v:
        if 'yes' == r[:3].lower():
            continue
        reasons.append(r[2:].strip())

reasons = '\n'.join(reasons)
prompt = 'All hallucination examples:'
prompt = '\n'.join([prompt, reasons])
prompt += '\n\nPlease summarize all the above exmaples and output different types of hallucinations line by line in bullets format:'

output = call(prompt, max_tokens=500)
print(output)

with open('../../data/musique_sample_summary_gpt-3.5.json', 'w') as f:
    f.write(output)


import time

import anthropic
import yaml

with open('../keys/anthropic_key', 'r') as f:
    api_key = f.read().strip()

client = anthropic.Anthropic(api_key=api_key)

def call(messages, model="claude-3-haiku-20240307", max_tokens=500):
    system_msg = ''
    if 'system' == messages[0]['role']:
        system_msg = messages[0]['content']
        messages = messages[1:]
    while True:
        try:
            response = client.messages.create(
                            model=model,
                            max_tokens=max_tokens,
                            temperature=0.2,
                            system=system_msg,
                            messages=messages,
                        )
            break
        except Exception as e:
            print('Errrrrrrrrrrrrrrrrrr', str(e))
            time.sleep(60)
    
    prediction = response.content[0].text

    return prediction
    
if __name__ == '__main__':
    pass


import tiktoken
import time
from openai import OpenAI

with open('../keys/openai_key', 'r') as f:
    api_key = f.readline().strip()

with open('../keys/openai_org_id', 'r') as f:
    organization = f.readline().strip()

client = OpenAI(api_key=api_key, organization=organization)

def call(message, model='gpt-3.5-turbo-0125', max_tokens=300):
    tokenizer = tiktoken.encoding_for_model(model)
    messages = [{'role': 'user', 'content': message}]

    while True:
        try:
            response = client.chat.completions.create(
                            model=model,
                            messages=messages,
                            max_tokens=max_tokens,
                            temperature=0.8,
                            top_p=0.9
                            )
            break
        except Exception as e:
            time.sleep(2)
            print('Errrrrrrrrrrrrrrrrrr', str(e))
            print(len(tokenizer.encode(messages[0]['content'])))

    prediction = response.choices[0].message.content

    return prediction

def call_chat(messages, model='gpt-3.5-turbo-0125', max_tokens=100):
    tokenizer = tiktoken.encoding_for_model(model)
    while True:
        try:
            response = client.chat.completions.create(
                            model=model,
                            messages=messages,
                            max_tokens=max_tokens,
                            temperature=0.8,
                            )
            break
        except Exception as e:
            time.sleep(2)
            print('Errrrrrrrrrrrrrrrrrr', str(e))
            print(len(tokenizer.encode(messages[0]['content'])))

    prediction = response.choices[0].message.content

    return prediction

def call_batch(batch_file_name):
    ### send file to OpenAI ###
    batch_file = client.files.create(
        file=open(batch_file_name, 'rb'),
        purpose='batch',
    )
    print(batch_file)

    batch_job = client.batches.create(
        input_file_id=batch_file.id,
        endpoint="/v1/chat/completions",
        completion_window="24h"
    )
    print(batch_job)

    return batch_job.id

def check_retrieve_batch(batch_id):
    ### check and try to retrieve files every 5 mins ###
    flag = False
    while True:
        batch_job = client.batches.retrieve(batch_id)
        print(batch_job.status)
        if 'completed' == batch_job.status:
            flag = True
            break
        elif batch_job.status in ['failed', 'expired', 'cancelled', 'cancelling']:
            break
        
        time.sleep(300)

    if not flag:
        exit()
    
    result_file_id = batch_job.output_file_id
    if result_file_id is None:
        error_file_id = batch_job.error_file_id
        results = client.files.content(error_file_id).content
    else:
        results = client.files.content(result_file_id).content

    return results


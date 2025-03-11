import json
import torch
import transformers
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from tqdm import tqdm

#llama_path = '/data/models/huggingface-format/llama-3-8b-instruct/'
llama_path = '/data/models/huggingface-format/llama-3-70b-instruct/'

#tokenizer = AutoTokenizer.from_pretrained(llama_path)
pipeline = transformers.pipeline(
    "text-generation",
    model=llama_path,
    torch_dtype=torch.float16,
    device_map="auto",
)
pipeline.tokenizer.pad_token_id = pipeline.tokenizer.eos_token_id
pipeline.tokenizer.padding_side = 'left'

terminators = [
    pipeline.tokenizer.eos_token_id,
    pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
]

def call(message, max_tokens=300):
    messages = [{'role': 'user', 'content': message}]
    prompt = pipeline.tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    with torch.no_grad():
        sequences = pipeline(
            prompt,
            do_sample=True,
            num_return_sequences=1,
            eos_token_id=terminators,
            pad_token_id=pipeline.tokenizer.eos_token_id,
            max_new_tokens=max_tokens,
            temperature=1,
        )
        # print("sequences: ", sequences)
        response = sequences[0]['generated_text'][len(prompt):]
    
    return response

class TEMP(Dataset):
    def __init__(self, prompts):
        self.prompts = []
        if isinstance(prompts[0], str):
            for prompt in prompts:
                self.prompts.append([{'role': 'user', 'content': prompt}])
        elif isinstance(prompts[0], tuple):
            for prompt in prompts:
                self.prompts.append([{'role': 'system', 'content': prompt[0]},
                                     {'role': 'user', 'content': prompt[1]}])
        self.prompts = pipeline.tokenizer.apply_chat_template(
            self.prompts,
            tokenize=False,
            add_generation_prompt=True
        )

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        return self.prompts[idx]
 
def batch_call(
        prompts,
        max_tokens=300,
        batch_size=4,
        save_path=None
        ):

    tmp_dataset = TEMP(prompts)
    outputs = pipeline(
        tmp_dataset,
        batch_size=batch_size,
        do_sample=True,
        num_return_sequences=1,
        eos_token_id=terminators,
        pad_token_id=pipeline.tokenizer.eos_token_id,
        max_new_tokens=max_tokens,
        temperature=1,
    )

    if save_path is not None:
        fp = open(save_path, 'a')

    responses = []
    for out, data in tqdm(zip(outputs, tmp_dataset), total=len(tmp_dataset)):
        response = out[0]['generated_text'][len(data):].strip()
        responses.append(response)
        if save_path is not None:
            fp.write(json.dumps({'input': data, 'output': response})+'\n')

    if save_path is not None:
        fp.close()
    return responses

if __name__ == '__main__':
    print('call:\n', call('hello'))
    print('batch_call:\n', batch_call(['hello', 'Mello']))

    text = """
The journey of artificial intelligence (AI) from its conceptual beginnings to its current status as a cornerstone of modern technology is a fascinating story of innovation, ambition, and, increasingly, introspection about the ethical dimensions of autonomous systems.

1. The Origins and Early Concepts of AI

Artificial intelligence as a formal field of academic study began in the mid-20th century, but the fascination with creating artifacts that could mimic human intelligence dates back much further. The early 1950s marked the foundational years, with pioneers such as Alan Turing, who proposed the Turing Test as a measure of machine intelligence. This test revolves around the idea that if a machine could converse with a human without the human realizing they were interacting with a machine, it could be considered "intelligent."

2. The Development Era

Throughout the 1960s and 1970s, the field of AI saw the development of the first neural networks and machine learning algorithms. Researchers at places like the Stanford AI Lab under the direction of John McCarthy, who coined the term "artificial intelligence," made significant strides. However, these early AI systems were limited by the technology of the time and often fell short of the grand expectations they inspired.

3. The AI Winter and Its Lessons

The late 1970s and 1980s experienced what is known as the "AI winter," a period when funding and interest in AI research temporarily waned due to the disillusionment with the inflated promises of AI capabilities. This period taught a valuable lesson about the hype and realistic progression in technology innovation, emphasizing the importance of incremental, sustainable advances.

4. The Resurgence and Modern AI

The resurgence of interest in AI began in the late 1990s and early 2000s, fueled by advances in computer power and data availability. This era saw the development of more sophisticated algorithms and the rise of deep learning, which allowed machines to process and learn from vast amounts of data in ways that were previously impossible. Projects like IBM’s Watson, which famously won the "Jeopardy!" game show, brought AI back into the public eye.

5. Current Applications

Today, AI permeates many aspects of life and industry. It drives the algorithms that recommend videos on YouTube, powers the voice assistants in smartphones, and aids in complex decision-making processes in sectors ranging from healthcare to finance. Autonomous vehicles, personalized medicine, and smart cities are just a few areas where AI is making a significant impact.

6. Ethical Considerations and Future Prospects

As AI technology advances, ethical considerations have become increasingly important. Issues such as privacy, surveillance, bias in AI algorithms, and the displacement of jobs due to automation are hot topics among policymakers, technologists, and the general public. Looking forward, the field is not just focused on making AI more powerful, but also on making it responsibly aligned with human values and societal needs.

7. The Next Frontier

The future of AI is likely to be marked by its convergence with other cutting-edge technologies like quantum computing and biotechnology. This convergence could potentially lead to breakthroughs in solving some of the world's most pressing challenges, such as climate change and complex diseases, as well as in understanding human cognition itself.

Conclusion

The evolution of AI is a testament to human ingenuity and a reflection of societal values and challenges. As we stand on the brink of what many consider to be a new era of AI, it is essential to guide this powerful technology with careful thought, considering both its immense potential and its significant risks.
    """
    print('call:\n', call(text))
    print('batch_call:\n', batch_call([text, text, text, text, text, text]))


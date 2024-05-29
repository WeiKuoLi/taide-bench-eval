import asyncio
import json
import os

from datasets import load_dataset
from transformers import AutoTokenizer
import openai 
from tqdm import tqdm
import torch
import argparse
from openai import AsyncOpenAI

def auto_load_dataset(dataset_path: str, dataset_split: str = 'train'):
    if dataset_path.endswith('.csv'):
        return load_dataset('csv', data_files=dataset_path)[dataset_split]
    return load_dataset(dataset_path)[dataset_split]


def add_template_to_instruction(inst: str):
    insts = [{ 'role': 'system', 'content': '你是一個說台灣繁體中文的AI助理。'}, # you can add system prompt here
                                            {'role': 'user', 'content': inst}
                                            ]
    '''
    result = tokenizer.apply_chat_template(insts, 
                                           tokenize=False,
                                           add_generation_prompt=True)'''
    return {'prompt': insts}

 
 
async def run_openai_inference(prompt, model='gpt-3.5-turbo', **kwargs):
    #client = AsyncOpenAI() 
    client = AsyncOpenAI()
    for _ in range(3):
        try:
            response = await client.chat.completions.create(
            #response = await client.chat.completions.async_create(
                model=model,
                messages=prompt,
                temperature=0.2,
                timeout=180,
                seed=42
            )
            return response.choices[0].message.content
        except Exception as e:
            # time.sleep(1)
            print(e)
            await asyncio.sleep(10)
    return ""
     


@torch.inference_mode()
async def main(
    openai_api_key: str,
    output_dir: str,
    tokenizer_path: str = 'google/mt5-small',
    dataset_path: str = 'taide/taide-bench',
    tasks: list[str] = ['summary',
                         'en2zh',
                         'zh2en',
                         'letter',
                         'essay', 
                         ],
    batch_size: int = 4,
    use_fast: bool = True,
    **kwargs
):
    print('openai_api_key', openai_api_key[:3]+'***')
    print('output_dir', output_dir)
    print('tokenizer_path', tokenizer_path)
    print('tasks', tasks)

    os.makedirs(output_dir, exist_ok=True)
    openai.api_key = openai_api_key

    #tokenizer_path = tokenizer_path
    #tokenizer = AutoTokenizer.from_pretrained(
    #    tokenizer_path,
    #    use_fast=use_fast
    #)
    #print(tokenizer.pad_token, tokenizer.eos_token)
    #tokenizer.pad_token = tokenizer.eos_token

    for task in tqdm(tasks):
        if os.path.exists(f'{output_dir}/resp_{task}.jsonl'):
            continue
        print(f'Generating for {task}')
        dataset = load_dataset(dataset_path, task)['train']
        dataset = dataset.map(
            lambda x: add_template_to_instruction(
                x['prompt']),
            desc='Adding template to prompt')

        result = []
        for i in tqdm(range(0, len(dataset), batch_size)):
            batch = dataset[i : i + batch_size]
            #print(batch)
            prompts = batch['prompt']

            tasks = [run_openai_inference(prompt, max_tokens=kwargs.pop('max_new_tokens', 512), temperature=0.7, **kwargs) for prompt in prompts]
            batch_results = await asyncio.gather(*tasks)

            result.extend([{'generated_text': text} for text in batch_results])
            #break
        print(result[:4])
        assert len(result) == len(dataset)

        output_path = f'{output_dir}/resp_{task}.jsonl'
        with open(output_path, 'w') as f:
            for r, x in zip(result, dataset):
                dct = {
                    'qid': x['qid'],
                    'model': 'gpt-3.5-turbo',  # Model used
                    'prompt': x['prompt'],
                    'resp': r['generated_text'],
                }
                x = {k: v for k, v in x.items() if isinstance(v, str)}
                x.update(dct)

                f.write(
                    json.dumps(x, ensure_ascii=False) + '\n'
                )



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("openai_api_key", type=str, help="OpenAI API Key")
    parser.add_argument("output_path", type=str, help="Output Path")
    parser.add_argument("--tasks", nargs='+', type=str, help="List of tasks")

    args = parser.parse_args()
    asyncio.run(main(args.openai_api_key, args.output_path, tasks=args.tasks))

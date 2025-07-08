# code to check if "yes"/"no" gold label rots ENTAIL the background
# additionally "neutral" NOT ENTAIL the background
import pandas as pd
import numpy as np
import random
from openai import OpenAI
import openai
import torch
import argparse
import os
from transformers import AutoTokenizer
from tqdm import tqdm
import time

random.seed(10)

with open('key/key.txt','r') as f:
    OPENAI_API_KEY = f.read().strip()
    
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

task_prompt = """Background:
{background}

Rule-of-thumb: {rot}

Is the above rule-of-thumb entailed by the background? Answer with Yes or No
"""


print(f"Device = {DEVICE}")
model_names = {
    "gpt3": "gpt-3.5-turbo-0125",
    "gpt4": "gpt-4-0613"
}
model_fn = {
    "llama2-7b": {'fn': "run_llm", 'name': "meta-llama/Llama-2-7b-hf", 'cache_path': "/data/tir/projects/tir5/users/ayerukol/hf_home/"},
    "llama2-7b-chat": {'fn': "run_llm", 'name': "meta-llama/Llama-2-7b-chat-hf", 'cache_path': "/data/datasets/models/huggingface/meta-llama/Llama-2-7b-chat-hf/"},
    "llama2-13b": {'fn': "run_vllm", 'name': "meta-llama/Llama-2-13b-hf", 'cache_path': "/data/tir/projects/tir5/users/ayerukol/hf_home/"},
    "llama2-13b-chat": {'fn': "run_llm", 'name': "meta-llama/Llama-2-13b-chat-hf", 'cache_path': "/data/datasets/models/huggingface/meta-llama/Llama-2-13b-chat-hf/"},
    "llama2-70b": {'fn': "run_vllm", 'name': "meta-llama/Llama-2-70b-hf", 'cache_path': "/data/datasets/models/huggingface/meta-llama/Llama-2-70b-hf/"},
    "llama2-70b-chat": {'fn': "run_llm", 'name': "meta-llama/Llama-2-70b-chat-hf", 'cache_path': "/data/datasets/models/huggingface/meta-llama/Llama-2-70b-chat-hf/"},
    "google/flan-t5-large": {'fn': "run_hf", 'cache_path': "/data/tir/projects/tir5/users/ayerukol/hf_home/"},
    "google/flan-t5-base":{'fn': "run_hf", 'cache_path': "/data/tir/projects/tir5/users/ayerukol/hf_home/"},
    "google/flan-t5-xl": {'fn': "run_hf", 'cache_path': "/data/tir/projects/tir5/users/ayerukol/hf_home/"},
    "google/flan-t5-xxl": {'fn': "run_hf", 'cache_path': "/data/tir/projects/tir5/users/ayerukol/hf_home/"},
    "mistral-chat": {'fn': "run_vllm", 'name': "mistralai/Mistral-7B-Instruct-v0.2", 'cache_path': "/data/tir/projects/tir5/users/ayerukol/vllm_home/"},
    "mistral-base": {'fn': "run_vllm", 'name': "mistralai/Mistral-7B-v0.1", 'cache_path': "/data/tir/projects/tir5/users/ayerukol/vllm_home/"},
    "zephyr-chat": {'fn': "run_llm", "name": "HuggingFaceH4/zephyr-7b-beta", 'cache_path': "/data/tir/projects/tir5/users/ayerukol/hf_home/"},
    "falcon-chat": {'fn': 'run_llm', "name": "tiiuae/falcon-7b-instruct", "cache_path": "/data/tir/projects/tir5/users/ayerukol/hf_home/"}
}

def get_completion(client, model, prompt, stream, temperature=1):
    response_json = client.completions.create(
            model=model,
            prompt=prompt,
            echo=False,
            n=1,
            stream=stream,
            #logprobs=1 if "gpt" not in model else None,
            max_tokens=50,
            temperature=temperature
        )
    return response_json.choices[0].text

def get_chat_completion(client, model, prompt, stream, temperature=1):
    response_json = client.chat.completions.create(
            model=model,
            messages=[
                {
                "role": "user",
                "content": prompt
                }
            ],
            
            n=1,
            stream=stream,
            #logprobs=1 if "gpt" not in model else None,
            max_tokens=30,
            temperature=temperature
        )
    return response_json.choices[0].message.content.split("\n\n")[0]


def get_responses(data, model, generation_type='chat', temperature_list=[0.0, 0.3, 1.0], conditioning='rot'):
    if "gpt" in model:
        client = OpenAI(
            api_key = OPENAI_API_KEY
        )
        # add model name for gpt styles models
        model = model_names[model]
        model_id = model
    else:
        url_dict = {
        "llama2-7b-chat": "http://localhost:8003/v1",
        "llama2-13b-chat": "http://localhost:8002/v1",
        "llama2-70b-chat": "http://localhost:8001/v1",
        "mistral-chat": "http://localhost:8000/v1",
        "zephyr-chat": "http://localhost:8004/v1",
        }
        client = OpenAI(
            api_key = "EMPTY",
            base_url = url_dict[model]
        )
        models = client.models.list()
        model_id = models.data[0].id
        if generation_type == "chat":
            if "meta-llama" in model_fn[model]['cache_path']:
                tokenizer = AutoTokenizer.from_pretrained(model_fn[model]['name'])
            else:
                tokenizer = AutoTokenizer.from_pretrained(model_fn[model]['name'], cache_dir=model_fn[model]['cache_path'])
        
    
    stream = False
    results_df = pd.DataFrame()
    
    # generate response for context, utterance
    for i in tqdm(range(len(data))):
        if i%100 == 0:
            time.sleep(1)
        
        ground_truth_prompt = task_prompt.format(background=data.loc[i, "Background"], rot=data.loc[i, 'Rule-of-Thumb'])
        

        for temp in temperature_list:
            if generation_type == "chat":
                gt_response = get_chat_completion(client, model_id, ground_truth_prompt, stream, temperature=temp)
                
            elif generation_type == "completion":
                gt_response = get_completion(client, model_id, ground_truth_prompt, stream, temperature=temp)
            
            gt_response = gt_response.lower()
            if data.loc[i, 'Gold Label'] in ['yes', 'no']:
                entail_pred =  int(gt_response == 'yes') # we want it to be 1, "yes"/"no" entail background
            elif data.loc[i, 'Gold Label'] == 'neutral':
                entail_pred = int(gt_response == "no") # we want it to be 1, "neutral" does not entail background
                     
            results_df.loc[i, "entail_pred_temp{}".format(str(temp))] = entail_pred
            
    results_df["model"] = model_id
    return results_df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, default="output/Etiquette_gpt4_3opt_irrel_try_postfilter_merged.csv")
    parser.add_argument("--output_path", type=str,  default="output/")
    parser.add_argument("--model", type=str, default="gpt4")
    parser.add_argument("--generation_type", type=str, default="chat")
    parser.add_argument("--temperature_list", nargs='+',  default="0.0",
                        type=lambda s: [float(item) for item in s.split(',')])
    
    args = parser.parse_args()
    print(args)
    
    input_data = pd.read_csv(args.input_path)
    input_data.fillna('', inplace=True)
    
    results_df = get_responses(input_data, args.model, generation_type=args.generation_type, temperature_list=args.temperature_list)
    if not os.path.exists(os.path.join(args.output_path, "stage1_rot")):
        os.makedirs(os.path.join(args.output_path, "stage1_rot"))
    results_df.fillna('', inplace=True)

    if results_df is not None:
        output_df = pd.concat([input_data, results_df], axis=1)
        output_df.to_csv(os.path.join(args.output_path, "stage1_rot", "data_with_rot_entailment.csv"))
        print("done!")


if __name__ == "__main__":
    main()
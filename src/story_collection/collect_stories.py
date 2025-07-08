from openai import OpenAI
import os
import json
import hydra
from omegaconf import DictConfig, OmegaConf
from typing import List, Dict, Tuple, Any
import logging
from tqdm import tqdm
from tqdm.asyncio import tqdm_asyncio
from time import sleep
from .few_shots_v3 import few_shots_gpt4_affirm, few_shots_gpt4_no, few_shots_gpt4_irrel
from src.story_collection.utils import PromptBuilder
import pandas as pd
import pdb
import random
import asyncio
import concurrent.futures
# from src.base import *

# TODO. the inglehart welzel cultural map is too biased. we should bin these differently.
map_iw_bin_to_country = {
    'African-Islamic': ['afghanistan', 'bangladesh','egypt','ethiopia','iran','iraq', 'kenya', 'lebanon', 'mauritius', 'pakistan','palestinian_territories','saudi_arabia', 'somalia', 'south_africa','south_sudan','sudan','syria', 'türkiye','zimbabwe'], # mauritius doesn't exist
    'Orthodox Europe': ['bosnia_and_herzegovina','greece','north_macedonia','romania','russia', 'serbia','ukraine'],
    'Confucian': ['china','hong_kong','japan','south_korea','taiwan'],
    'Catholic Europe': ['austria','croatia','hungary','italy','poland','portugal','spain'],
    'Protestant Europe': ['france','germany','netherlands','sweden',],
    'English Speaking': ['australia','canada','ireland','new_zealand','united_kingdom','united_states_of_america'],
    'Latin America': ['argentina','brazil','chile','colombia','mexico', 'malta','peru','philippines','venezuela'], # apparently malta goes here, and philippines????
    'West and South Asia': ['cambodia','cyprus','fiji','india','indonesia','israel','laos','malaysia','myanmar','nepal','papua_new_guinea','samoa','singapore','sri_lanka','thailand', 'timor-leste', 'tonga', 'vietnam'] # papua new guinea, samoa, timor-leste, tonga
}

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

COLOR_RED = '\033[91m'
COLOR_GREEN = '\033[92m'
COLOR_BLUE = '\033[94m'
COLOR_NORMAL = '\033[0m'

def get_cultcontext(cfg: DictConfig, axis_name: str) -> Dict[str, List[Tuple[str,str]]]:
    # get the data directory from cfg
    data_dir = cfg.webscrape.data_dir

    axis_dir = os.path.join(data_dir, axis_name)

    files = os.listdir(axis_dir)

    countries = ['_'.join(file.split('_')[:-1]) for file in files]

    content: Dict[str, List[Tuple[str, str]]] = {}

    for file, country in zip(files, countries):
        with open(os.path.join(axis_dir, file), 'r') as f:
            # check if json and parse
            if file.endswith('.json'):
                content_dict = json.load(f)
                # get the content, and split along subaxis (eg. Basic Etiquette / Dining for axis = Etiquette)
                content['_'.join(file.split('_')[:-1])] = []
                for subaxis, value in content_dict['content'].items():
                    content_str = f"### {subaxis}\n"
                    content_list = []
                    for val in value:
                        content_str += f"- {val}\n"
                        content_list.append(val)
                    subaxis_ = '_'.join(subaxis.lower().strip().split())
                    content[country].append((subaxis_, content_str, content_list))
    return content

def few_shot_dump(cfg: DictConfig, system_ast_prompt: List[Dict], append=True):
    assert type(system_ast_prompt) is list, "prompt needs to be a list of atleast 1 value"
    assert type(system_ast_prompt[0]) is dict, "prompt structure: must be a dict of system and assistant responses"

    file_mode = 'a' if append else 'w'
    with open(f'{os.path.join(cfg.prompts.prompts_dir, "few_shots.txt")}',file_mode) as outfile:
        for sap in system_ast_prompt:
            outfile.write(json.dumps(sap)+'\n')

def parse_csv(prompt: str, id: int, content: str, country: str, other_country: str, axis: str, subaxis: str, norm: str, other_norm: str, country_background: str, other_background: str, label: str):
    
    if label != 'neutral':
        value_str = 'value:'
        rule_str = 'rule-of-thumb:'
    else:
        value_str = 'irrelevant value:'
        rule_str = 'irrelevant rule-of-thumb:'
    
    story_str = 'story:'
    explanation_str = 'explanation:'
    
    value_index = content.lower().rindex(value_str)+len(value_str)
    rule_index = content.lower().rindex(rule_str)+len(rule_str)
    story_index = content.lower().rindex(story_str)+len(story_str)
    explanation_index = content.lower().rindex(explanation_str)+len(explanation_str)
    
    # get the values, and strip any ##
    value = content[value_index:rule_index-len(rule_str)].strip().strip('#').strip() if rule_index > value_index else content[value_index:].strip().strip('#').strip()
    rule = content[rule_index:story_index-len(story_str)].strip().strip('#').strip() if story_index > rule_index else content[rule_index:].strip().strip('#').strip()
    story = content[story_index:explanation_index-len(explanation_str)].strip().strip('#').strip() if explanation_index > story_index else content[story_index:].strip().strip('#').strip()
    explanation = content[explanation_index:].strip().strip('#').strip() if explanation_index > story_index else ""
    # add to a json
    return {'ID': id, 
            'Country': country, 
            'Other Country': other_country,
            'Background': country_background, 
            'Other Background': other_background,
            'Norm': norm,
            'Other Norm': other_norm,
            'Axis': axis ,
            'Subaxis': subaxis, 
            'Value': value, 
            'Rule-of-Thumb': rule, 
            'Story': story, 
            'Explanation': explanation, 
            'Gold Label': label}

def get_iw_bin(country: str, map_iw_bin_to_country: Dict[str, List[str]]) -> str:
    for iw_bin, countries in map_iw_bin_to_country.items():
        if country in countries:
            return iw_bin
    return None

def fetch_response(client, prompt, story_counter_id, country, other_country, axis_name, subaxis, norm, other_norm, cultcontext, other_cultcontext, label):
        try:
            response = client.chat.completions.create(model="gpt-4-turbo-preview", 
                                                            messages=prompt, 
                                                            max_tokens=1024, 
                                                            temperature=0.9, 
                                                            timeout=60)
            completion = response.choices[0]
            sleep(1)

            return parse_csv(prompt, story_counter_id, completion.message.content, country, other_country, axis_name, subaxis, norm, other_norm, cultcontext, other_cultcontext, label)
        
        except Exception as e:
            print("ERROR!!!")
            logger.error(f"Error in {country} for {axis_name} for label: {label}")
            logger.error(f"{e}")
            
            with open('bad_prompt_collections.txt', 'a') as outfile:
                outfile.write(f"{country}\t{axis_name}\t{subaxis}\t{label}\n")
            return None

import functools # adding kwargs for easier passing
async def run_predictions(client, arg_list):
    loop = asyncio.get_running_loop()
    with concurrent.futures.ThreadPoolExecutor() as pool:
        results = await tqdm_asyncio.gather(
            *[
                loop.run_in_executor(
                    pool,
                    functools.partial(fetch_response,
                    client, **args)
                ) 
                for args in arg_list
            ]
        )
    return results      

@hydra.main(version_base=None, config_name="config", config_path="../../conf")
def main(cfg: DictConfig) -> None:
    print("Please verify config values")
    print(OmegaConf.to_yaml(cfg))
    user_input = input("Continue? [y/n] [N] ")
    if user_input.strip().lower() != 'y':
        print("Exiting...")
        exit()
    
    with open(f"{cfg.prompts.openai_key}", 'r') as f:
        OPENAI_API_KEY = f.read().strip()

    client = OpenAI(api_key=OPENAI_API_KEY)

    axis_name = 'Etiquette'
    contexts = get_cultcontext(cfg=cfg, axis_name=axis_name)

    start_index = cfg.prompts.start_index if cfg.prompts.start_index else 0
    end_index = cfg.prompts.end_index if cfg.prompts.end_index else len(contexts)
    
    promptBuilder = PromptBuilder(model_name='gpt-4-turbo')

    
    responses = []
    context_items = list(contexts.items())[start_index:end_index]
    label2fewshot = {
        'yes': few_shots_gpt4_affirm,
        'no': few_shots_gpt4_no,
        'neutral': few_shots_gpt4_irrel
    }

    story_counter_id = 0
    tasks = []


    names_mapping =  {
                'basic_etiquette': 'basic_etiquette', 
                'manners_in_vietnam': 'basic_etiquette',
                'māori_etiquette': 'basic_etiquette',
                'cleanliness': 'basic_etiquette',
                'direct_manners': 'basic_etiquette',
                'tipping': 'basic_etiquette', 
                "‘taarof’_(politeness_and_mutual_respect)": 'basic_etiquette',
                'pub_etiquette': 'basic_etiquette',
                 'visiting': 'visiting', 
                 'visiting_and_eating': 'visiting',
                 'visiting_a_village': 'visiting',
                 'eating': 'eating', 
                 'eating_out': 'eating',
                 'religious_dietary_laws': 'eating',
                 'drinking': 'eating',
                 'drinking_coffee': 'eating',
                 'toasting': 'eating',
                 'gifts': 'gifts',
                 'gift-giving': 'gifts',
                 'gift_giving': 'gifts',
                 'offering_and_complimenting_items': 'gifts',
             }

    count = 0
    for i in range(cfg.prompts.num_story_samples):
        for label in ['yes', 'no', 'neutral']:
            for country, cultcontext_with_subaxis in context_items:
                
                for subaxis, cultcontext, cult_list in cultcontext_with_subaxis:
                    count += 1
                    subaxis_ = names_mapping[subaxis]

                    random_norm = random.choice(cult_list)
                    cult = f"\n---\n### Country:\n{country}\n\n### Cultural background [{axis_name}]:\n{random_norm}"
                    other_country = country
                    other_cultcontext = cultcontext
                    other_random_norm = random_norm

                    if label == 'neutral':
                        iw_bin = get_iw_bin(country, map_iw_bin_to_country)
                        other_country_bin = random.choice(list(map_iw_bin_to_country.keys() - {iw_bin}))
                        other_country = random.choice(map_iw_bin_to_country[other_country_bin])

                        if subaxis_ in [names_mapping[sub] for sub, _, _ in contexts[other_country]]:
                            other_cultcontext = [cultcontext for sub, cultcontext, cult_list in contexts[other_country] if names_mapping[sub] == subaxis_][0]
                            other_cultcontext_list = [cult_list for sub, cultcontext, cult_list in contexts[other_country] if names_mapping[sub] == subaxis_][0]
                        
                        else:
                            # sample from another axis as the current one is not present
                            sub, other_cultcontext, other_cultcontext_list = random.choice(contexts[other_country])
                       
                        other_random_norm = random.choice(other_cultcontext_list)
                        cult += f"\n---\n### Country:\n{other_country}\n\n### Cultural background [{axis_name}]:\n{other_random_norm}"

                    prompt = promptBuilder.story_generation_prompt_constructor(cfg=cfg, few_shots=label2fewshot[label], background=cult, label=label)
                    
                    tasks.append({'prompt': prompt, 
                                                'story_counter_id': story_counter_id, 
                                                'country': country, 
                                                'other_country': other_country, 
                                                'axis_name': axis_name, 
                                                'subaxis': subaxis_, 
                                                'cultcontext': cultcontext,
                                                'norm': random_norm, 
                                                'other_norm': other_random_norm, 
                                                'other_cultcontext': other_cultcontext,
                                                'label': label})
                    
                    story_counter_id += 1
    
    batch_size = 128
    results = []
    for pos in range(0, len(tasks), batch_size):
        task_batch = tasks[pos:pos + batch_size]
        result_batch = asyncio.run(run_predictions(client, task_batch))
        results.extend(result_batch)

    responses = [res for res in results if res is not None]

    df = pd.DataFrame(responses)
    if not os.path.isdir(cfg.prompts.output_dir):
        os.mkdir(cfg.prompts.output_dir)
    if not os.path.isdir(f"{cfg.prompts.output_dir}/{axis_name}"):
        os.mkdir(f"{cfg.prompts.output_dir}/{axis_name}")
    df.to_csv(os.path.join(cfg.prompts.output_dir, f"{axis_name}_gpt4_3opt.csv"), index=False)
    
    print(f"Saved responses to {os.path.join(cfg.prompts.output_dir, f'{axis_name}_gpt4_3opt.csv')}, length: {len(responses)}")



if __name__ == '__main__':
    main()
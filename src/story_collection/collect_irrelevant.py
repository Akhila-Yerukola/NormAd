from openai import OpenAI
import os
import json
import hydra
from omegaconf import DictConfig, OmegaConf
from typing import List, Dict, Tuple, Any
import logging
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm
from time import sleep
from .few_shots_v2 import few_shots_gpt4_affirm, few_shots_gpt4_no, few_shots_gpt4_irrel
from src.story_collection.utils import PromptBuilder
import pandas as pd
import pdb
import random

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
                    for val in value:
                        content_str += f"- {val}\n"
                    subaxis_ = '_'.join(subaxis.lower().strip().split())
                    content[country].append((subaxis_, content_str))
    return content

def few_shot_dump(cfg: DictConfig, system_ast_prompt: List[Dict], append=True):
    assert type(system_ast_prompt) is list, "prompt needs to be a list of atleast 1 value"
    assert type(system_ast_prompt[0]) is dict, "prompt structure: must be a dict of system and assistant responses"

    file_mode = 'a' if append else 'w'
    with open(f'{os.path.join(cfg.prompts.prompts_dir, "few_shots.txt")}',file_mode) as outfile:
        for sap in system_ast_prompt:
            outfile.write(json.dumps(sap)+'\n')

def parse_csv(id: int, content: str, country: str, other_country: str, axis: str, subaxis: str, country_background: str, other_country_background: str, label: str):
    if label != 'neutral':
        value_str = 'value:'
        rule_str = 'rule-of-thumb:'
    else:
        value_str = 'Value for Country 2 (Unrelated to Cultural Background for Country 1):'.lower()
        rule_str = 'Rule-of-Thumb for Country 2 (Unrelated to Cultural Background for Country 1):'.lower()
    story_str = 'Story based on Country 1 (Unrelated to Cultural Background for Country 2):'.lower()
    explanation_str = 'explanation:'
    
    value_index = content.lower().rindex(value_str)+len(value_str)
    rule_index = content.lower().rindex(rule_str)+len(rule_str)
    story_index = content.lower().rindex(story_str)+len(story_str)
    try:
        explanation_index = content.lower().rindex(explanation_str)+len(explanation_str)
    except:
        explanation_index = -1
    
    # get the values, and strip any ##
    try:
        value = content[value_index:rule_index-len(rule_str)].strip().strip('#').strip() if rule_index > value_index else content[value_index:].strip().strip('#').strip()
        rule = content[rule_index:].strip().split('\n')[0].strip().strip('#').strip()
        story = content[story_index:explanation_index-len(explanation_str)].strip().strip('#').strip() if explanation_index > story_index else content[story_index:].strip().strip('#').strip()
        explanation = content[explanation_index:].strip().strip('#').strip() if explanation_index > story_index else ""
    except:
        logger.error(f"{COLOR_RED}Error in parsing content for {country} for {axis} for label: {label}{COLOR_NORMAL}")
    # add to a json


    return {'ID': id, 'Country': country, 'Other Country': other_country, 'Background': country_background,'Other Background': other_country_background,'Axis': axis ,'Subaxis': subaxis, 'Value': value, 'Rule-of-Thumb': rule, 'Story': story, 'Explanation': explanation, 'Gold Label': label}

def get_iw_bin(country: str, map_iw_bin_to_country: Dict[str, List[str]]) -> str:
    for iw_bin, countries in map_iw_bin_to_country.items():
        if country in countries:
            return iw_bin
    return None
@hydra.main(version_base=None, config_name="config", config_path="../../conf")
def main(cfg: DictConfig) -> None:
    
    # ask user to check config values
    print(f"{COLOR_RED} Please verify config values {COLOR_BLUE}")
    print(OmegaConf.to_yaml(cfg))
    # user_input = input(f"{COLOR_GREEN}Continue? [y/n] [N] {COLOR_NORMAL}")
    # if user_input.strip().lower() != 'y':
    #     print(f"{COLOR_RED}Exiting...{COLOR_NORMAL}")
    #     exit()
    
    # read in the openai api key
    with open(f"{cfg.prompts.openai_key}", 'r') as f:
        OPENAI_API_KEY = f.read().strip()

    client = OpenAI(api_key=OPENAI_API_KEY)

    axis_name = 'Etiquette' # TODO for all axes
    contexts = get_cultcontext(cfg=cfg, axis_name=axis_name)

    start_index = cfg.prompts.start_index if cfg.prompts.start_index else 0
    end_index = cfg.prompts.end_index if cfg.prompts.end_index else len(contexts)
    
    promptBuilder = PromptBuilder(model_name='gpt-4-turbo')

    with logging_redirect_tqdm():
        # get context items with splices 
        responses = []
        context_items = list(contexts.items())[start_index:end_index]
        label2fewshot = {
                'neutral': few_shots_gpt4_irrel
            }
        story_counter_id = 0
        for label in tqdm(['neutral'], position=0, desc="Labels", colour='blue'):
            for country, cultcontext_with_subaxis in (country_bar := tqdm(context_items, desc=f"Axis: {axis_name}", colour='red', position=1)):
                country_bar.set_description(f"Country: {country}")
                for subaxis, cultcontext in (subaxis_bar := tqdm(cultcontext_with_subaxis, desc=f"Country: {country}", colour='green', position=2, leave=False)):
                    subaxis_bar.set_description(f"Cultural background [{axis_name}]: {subaxis}")
                    logger.info(f"{COLOR_GREEN}Country: {country}{COLOR_NORMAL}")
                    logger.info(f"{COLOR_GREEN}Cultural background [{axis_name}]: {subaxis}{COLOR_NORMAL}")
                    
                    # add the country name and the cultural axis to the prompt
                    cult = f"\n---\n### Country 1:\n{country}\n\n### Cultural background for Country 1 [{axis_name}]:\n{cultcontext}"
                    if label == 'neutral':
                        # sample another country from a different iw bin
                        iw_bin = get_iw_bin(country, map_iw_bin_to_country)
                        other_country_bin = random.choice(list(map_iw_bin_to_country.keys() - {iw_bin}))
                        other_country = random.choice(map_iw_bin_to_country[other_country_bin])
                        if subaxis in [sub for sub, _ in contexts[other_country]]:
                            other_cultcontext = [cultcontext for sub, cultcontext in contexts[other_country] if sub == subaxis][0]
                        else:
                            other_cultcontext = random.choice(contexts[other_country][1])
                        # add the country name and the cultural axis to the prompt
                        cult += f"\n---\n### Country 2:\n{other_country}\n\n### Cultural background for Country 2 [{axis_name}]:\n{other_cultcontext}"
                    prompt = promptBuilder.story_generation_prompt_constructor (cfg=cfg, few_shots=label2fewshot[label], cultcontext=cult, label=label)
                    logging.debug(f"Prompt Length: {len(prompt[0]['content'])}")
                    try:
                        response = client.chat.completions.create(model="gpt-4-turbo-preview", messages=prompt, max_tokens=1024, temperature=0.5, n = cfg.prompts.num_story_samples, timeout=60)
                        # save the response in output directory as country_name_{axis}.md
                        # if the directory doesn't exist, make it
                        if not os.path.isdir(f"{cfg.prompts.output_dir}"):
                            os.mkdir(f"{cfg.prompts.output_dir}")

                        # if the axis directory doesn't exist, make it
                        if not os.path.isdir(f"{cfg.prompts.output_dir}/{axis_name}"):
                            os.mkdir(f"{cfg.prompts.output_dir}/{axis_name}")
                        
                        for completion in response.choices:                                                   
                            responses.append(parse_csv(story_counter_id, completion.message.content, country, other_country, axis_name, subaxis, cultcontext, other_cultcontext, label))                        
                            story_counter_id += 1
                            logger.debug(completion.message.content)
                    
                    except Exception as e:
                        logger.error(f"{COLOR_RED}Error in {country} for {axis_name}{COLOR_NORMAL} for label: {label}")
                        logger.error(f"{COLOR_BLUE}{e}{COLOR_NORMAL}")
                        with open('bad_prompt_collections.txt','a') as outfile:
                            # dump the country, axis and subaxis for which there was a problem
                            outfile.write(f"{country}\t{axis_name}\t{subaxis}\t{label}\n")

            df = pd.DataFrame(responses)
            df.to_csv(os.path.join(cfg.prompts.output_dir, f"{axis_name}_gpt4_3opt_irrelevant.csv"), index=False)
            logger.debug(f"Saved responses to {os.path.join(cfg.prompts.output_dir, f'{axis_name}_gpt4.csv')}, length: {len(responses)}")

if __name__ == '__main__':
    main()

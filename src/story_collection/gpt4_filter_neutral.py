import os
import json
import hydra
from omegaconf import DictConfig, OmegaConf
from typing import List, Dict, Tuple, Any
from openai import OpenAI
from collections import defaultdict
import logging
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

import pandas as pd
import pdb
import random
from src.story_collection.collect_irrelevant import get_iw_bin, map_iw_bin_to_country, parse_csv, get_cultcontext
from src.story_collection.few_shots_v2 import few_shots_gpt4_affirm, few_shots_gpt4_no, few_shots_gpt4_irrel
from src.story_collection.utils import PromptBuilder

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

COLOR_RED = '\033[91m'
COLOR_GREEN = '\033[92m'
COLOR_BLUE = '\033[94m'
COLOR_NORMAL = '\033[0m'

label2fewshot = {
        # 'yes': few_shots_gpt4_affirm,
        # 'no': few_shots_gpt4_no,
        'neutral': few_shots_gpt4_irrel
    }

filter_prompt = [{'role': 'system', 'content':'''Rule-of-Thumb: One should politely decline an offer twice before accepting on the third time.
Story: During an evening out with friends, Henry immediately accepted when his friend offered to buy him a drink, without offering to purchase the next round for everyone. Is what Henry did socially acceptable?
Is this story relevant to the rule-of-thumb (Yes/No ONLY)?'''},{'role': 'assistant', 'content': 'Yes'},{'role': 'system', 'content':'''Rule-of-Thumb: It is polite to comment on the host's use of their finest crockery and cutlery out of acknowledgement for their hospitality.
Story: Alex decided to spontaneously visit Jamie's house late in the afternoon without giving any prior notice. Jamie was surprised to see Alex but welcomed him in. Is what Alex did socially acceptable?
Is this story relevant to the rule-of-thumb (Yes/No ONLY)?'''},{'role': 'assistant', 'content': 'No'}]


def get_specific_cultcontext(cfg: DictConfig, axis_name: str, req_subaxis: str, req_country: str) -> Dict[str, List[Tuple[str,str]]]:
    # get the data directory from cfg
    data_dir = cfg.webscrape.data_dir

    axis_dir = os.path.join(data_dir, axis_name)

    files = os.listdir(axis_dir)

    countries = ['_'.join(file.split('_')[:-1]) for file in files]

    content: Dict[str, List[Tuple[str, str]]] = {}

    for file, country in zip(files, countries):
        if country == req_country:
            with open(os.path.join(axis_dir, file), 'r') as f:
                # check if json and parse
                if file.endswith('.json'):
                    content_dict = json.load(f)
                    # get the content, and split along subaxis (eg. Basic Etiquette / Dining for axis = Etiquette)
                    content['_'.join(file.split('_')[:-1])] = []
                    for subaxis, value in content_dict['content'].items():
                        subaxis_ = '_'.join(subaxis.lower().strip().split())
                        if subaxis_ == req_subaxis:
                            content_str = f"### {subaxis}\n"
                            for val in value:
                                content_str += f"- {val}\n"
                            
                            return content_str

@hydra.main(version_base=None, config_name="config", config_path="../../conf")
def main(cfg: DictConfig) -> None:
    story_dir = cfg.prompts.output_dir
    story_path = cfg.story_inference.path_to_story_file
    #story_path = os.path.join(story_dir, story_file)
    with open(cfg.prompts.openai_key, 'r') as f:
        openai_key = f.read().strip()
    openai_client = OpenAI(api_key=openai_key)
    story_df = pd.read_csv(story_path)
    story_df_neutral = story_df[story_df['Gold Label'] == 'neutral']
    orig_len = story_df_neutral.shape[0]
    relevance = []

    init_counter = story_df_neutral.iloc[0].name
    for i, row in tqdm(story_df_neutral.iterrows(), total=story_df_neutral.shape[0]):
        rot = row['Rule-of-Thumb']
        story = row['Story']
        prompt = filter_prompt + [{'role': 'system', 'content': f'''Rule-of-Thumb: {rot}\nStory: {story}\nIs this story relevant to the rule-of-thumb (Yes/No ONLY)?'''}]
        response = openai_client.chat.completions.create(model="gpt-4", messages=prompt, max_tokens=1, temperature=0, timeout=60)
        message = response.choices[0].message.content
        relevance.append(False if message == 'No' else True)

    # get indices for relevance
    relevance_indices = [i for i, x in enumerate(relevance) if x]
    # get the country, axis and subaxis for the relevant stories
    story_rel = story_df_neutral.iloc[relevance_indices]
    story_non_rel = story_df_neutral.iloc[[i for i in range(story_df_neutral.shape[0]) if i not in relevance_indices]]
    pdb.set_trace()
    story_non_rel.to_csv('./output/Etiquette_gpt4_3opt_irrel_try_4filter.csv',index=False)
    country, axis, subaxis = story_rel['Country'], story_rel['Axis'], story_rel['Subaxis']
    
    # get stories for these again!

    promptBuilder = PromptBuilder(model_name='gpt-4-turbo')
    specific_contexts_stripped = []
    for c, a, s in zip(country, axis, subaxis):
        sc = get_specific_cultcontext(cfg, a, s, c)
        if sc is None:
            pdb.set_trace()
        specific_contexts_stripped.append((c, s, sc))

    axis_name = 'Etiquette'

    responses = []
    all_contexts = get_cultcontext(cfg, axis_name)
    story_counter_id = 0
    for country, subaxis, cultcontext in (subaxis_bar := tqdm(specific_contexts_stripped, colour='green', position=1, leave=False)):
        subaxis_bar.set_description(f"Cultural background [{axis_name}]: {subaxis}")
        logger.info(f"{COLOR_GREEN}Country: {country}{COLOR_NORMAL}")
        logger.info(f"{COLOR_GREEN}Cultural background [{axis_name}]: {subaxis}{COLOR_NORMAL}")
        
        # add the country name and the cultural axis to the prompt
        cult = f"\n---\n### Country 1:\n{country}\n\n### Cultural background for Country 1 [{axis_name}]:\n{cultcontext}"
        # sample another country from a different iw bin
        iw_bin = get_iw_bin(country, map_iw_bin_to_country)
        other_country_bin = random.choice(list(map_iw_bin_to_country.keys() - {iw_bin}))
        other_country = random.choice(map_iw_bin_to_country[other_country_bin])
        if subaxis in [sub for sub, _ in all_contexts[other_country]]:
            other_cultcontext = [cultcontext for sub, cultcontext in all_contexts[other_country] if sub == subaxis][0]
        else:
            other_cultcontext = random.choice(all_contexts[other_country][1])
        # add the country name and the cultural axis to the prompt
        cult += f"\n---\n### Country 2:\n{other_country}\n\n### Cultural background for Country 2 [{axis_name}]:\n{other_cultcontext}"
        prompt = promptBuilder.story_generation_prompt_constructor (cfg=cfg, few_shots=label2fewshot['neutral'], cultcontext=cult, label='neutral')
        logging.debug(f"Prompt Length: {len(prompt[0]['content'])}")
        try:
            response = openai_client.chat.completions.create(model="gpt-4-turbo-preview", messages=prompt, max_tokens=1024, temperature=0.9, n = 1, timeout=60)
            
            # save the response in output directory as country_name_{axis}.md
            # if the directory doesn't exist, make it
            if not os.path.isdir(f"{cfg.prompts.output_dir}"):
                os.mkdir(f"{cfg.prompts.output_dir}")

            # if the axis directory doesn't exist, make it
            if not os.path.isdir(f"{cfg.prompts.output_dir}/{axis_name}"):
                os.mkdir(f"{cfg.prompts.output_dir}/{axis_name}")
            
            for completion in response.choices:
                responses.append(parse_csv(id=story_counter_id, content=completion.message.content, country=country, axis=axis_name, subaxis=subaxis, country_background=cultcontext, label='neutral', other_country=other_country, other_country_background=other_cultcontext))                        
                story_counter_id += 1
                logger.debug(completion.message.content)
        
        except Exception as e:
            logger.error(f"{COLOR_RED}Error in {country} for {axis_name}{COLOR_NORMAL} for label: {'neutral'}")
            logger.error(f"{COLOR_BLUE}{e}{COLOR_NORMAL}")
            with open('bad_prompt_collections.txt','a') as outfile:
                # dump the country, axis and subaxis for which there was a problem
                outfile.write(f"{country}\t{axis_name}\t{subaxis}\t{'neutral'}\n")

    # set the responses to the story_df_neutral at each relevance index
    for i, response in enumerate(responses):
        story_df_neutral.iloc[relevance_indices[i]] = response

    story_df_non_neutral = story_df[story_df['Gold Label'] != 'neutral']
    story_df_full = pd.concat((story_df_non_neutral, story_df_neutral))
    story_df_full['ID'] = [i for i in range(len(story_df_full))]
    story_df_full.to_csv('./output/Etiquette_gpt4_3opt_irrel_try_3filter.csv',index=False)
    # # refilter the responses
    # return 0
    # relevance = []
    # for i, row in tqdm(story_df_neutral.iterrows(), total=story_df_neutral.shape[0]):
    #     rot = row['Rule-of-Thumb']
    #     story = row['Story']
    #     prompt = filter_prompt + [{'role': 'system', 'content': f'''Rule-of-Thumb: {rot}\nStory: {story}\nIs this story relevant to the rule-of-thumb (Yes/No ONLY)?'''}]
    #     response = openai_client.chat.completions.create(model="gpt-4", messages=prompt, max_tokens=1, temperature=0, timeout=60)
    #     message = response.choices[0].message.content
    #     relevance.append(False if message == 'No' else True)


    # # print number filtered now
    # logger.info(f"Number of stories filtered: {orig_len - sum(relevance)}")





main()


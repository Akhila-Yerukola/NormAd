# get response after sampling a random country from the IWSLT bins
import pandas as pd

import pdb

import pandas as pd
from rlkf.src.model.openai_model import OpenAIInferencer
import hydra
import logging
from typing import List, Dict
from omegaconf import DictConfig, OmegaConf
import pandas as pd
import os
import pdb
import random
from tqdm import tqdm
from src.response_collection.utils import map_conditioning_to_attr

COLOR_RED = '\033[91m'
COLOR_GREEN = '\033[92m'
COLOR_BLUE = '\033[94m'
COLOR_NORMAL = '\033[0m'

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


def preprocess_story(story_df: pd.DataFrame) -> pd.DataFrame:
    # remove rejected stories
    story_df = story_df[story_df['Reject Story'].apply(lambda x: x.lower() if type(x) == str else x) != 'y']
    # replace story with corrected stories if exists
    def correct_story(x: Dict) -> Dict:
        if pd.isna(x['Corrected Story']):
            x['Story'] = x['Story']
        else:
            x['Story'] = x['Corrected Story']
        return x
    
    viable_group_names = {'basic_etiquette': 'basic_etiquette', 
                'visiting': 'visiting', 
                'eating': 'eating', 
                'gifts': 'gifts',
                'gift_giving': 'gifts'}
        
    def normalize_column_names(x: Dict) -> Dict:
        if x['Subaxis'] not in viable_group_names.keys():
            logger.debug(f"{COLOR_RED}Subaxis {x['Subaxis']} not in viable group names{COLOR_NORMAL}")
            return x
        for key in viable_group_names.keys():
            if key in x['Subaxis'].lower():
                x['Subaxis'] = viable_group_names[key]
                break
        return x
        
    story_df = story_df[story_df['Subaxis'].isin(viable_group_names.keys())]
    story_df = story_df.apply(normalize_column_names, axis=1)
    story_df = story_df.apply(correct_story, axis=1)
    story_df.drop('Corrected Story', axis=1, inplace=True)
    story_df.drop('Reject Story', axis=1, inplace=True) 
    story_df.drop('Comments', axis=1, inplace=True)

    return story_df


def sample_iwslt_bin(story_df: pd.DataFrame):
    iwlist_df = pd.read_csv('src/analysis/country_iwlist.csv')
    iwslt_bins = iwlist_df['iw_bin'].unique()
    story_df = preprocess_story(story_df=story_df)
    # for every row, sample a value, rot combination from countries in all other bin
    # store these in a new dataframe
    sampled_story_df = []
    for ind, row in tqdm(story_df.iterrows()):
        # find the bin of the country
        country = row['Country']
        country_bin = iwlist_df[iwlist_df['country'] == country]['iw_bin'].values[0]
        # sample a country from all other bins
        for bin in iwslt_bins:
            if bin == country_bin:
                continue
            else:
                try: 
                    list_of_countries = iwlist_df[iwlist_df['iw_bin'] == bin]['country'].tolist()
                    #shuffle
                    random.shuffle(list_of_countries)
                    for sampled_country in list_of_countries:
                        # sample a value, rot, background from the sampled country with the same subaxis as original
                        sampled_row = story_df[(story_df['Country'] == sampled_country)]
                        # logger.debug("Sampled row: ")
                        # logger.debug(sampled_row[['Country', 'Subaxis']])
                        sampled_row = sampled_row[sampled_row['Subaxis'] == row['Subaxis']]
                        if sampled_row.empty:
                            # logger.info(f"{COLOR_RED}Sampled row is empty for {sampled_country} and subaxis {row['Subaxis']}{COLOR_NORMAL}")
                            continue
                        else:
                            sampled_row = sampled_row.sample(1).iloc[0]
                            break
                    if sampled_row.empty:
                        logger.info(f"No sampled row found for {country} and subaxis {row['Subaxis']}")
                        raise Exception
                except Exception as e:
                    logger.error(f"{COLOR_RED}Error in sampling from bin {bin} for country {country}{COLOR_NORMAL}")
                    pdb.set_trace()
                    continue
                sampled_value = sampled_row['Value']
                sampled_rot = sampled_row['Rule-of-Thumb']
                sampled_background = sampled_row['Background']
                # create a new row with the sampled value and rot
                new_row = row.copy()
                new_row['Country'], new_row['Value'], new_row['Rule-of-Thumb'], new_row['Background'] = sampled_country, sampled_value, sampled_rot, sampled_background
                # add the original value and rot to the new row
                new_row['Original Country'], new_row['Original Value'], new_row['Original Rule-of-Thumb'], new_row['Original Background'] = row['Country'], row['Value'], row['Rule-of-Thumb'], row['Background']
                # append to the new dataframe
                sampled_story_df.append(new_row)
        # add original row to the dataframe as well after adding the Original columns
        row['Original Country'] = row['Country']
        row['Original Value'] = row['Value']
        row['Original Rule-of-Thumb'] = row['Rule-of-Thumb']
        row['Original Background'] = row['Background']
        sampled_story_df.append(row)

    sampled_story_df = pd.DataFrame(sampled_story_df)
    return sampled_story_df
        
def postprocess_dataframe(cfg: DictConfig, story_df: pd.DataFrame) -> pd.DataFrame:
    # Find the original response for every row in the dataframe
    for ind, row in story_df.iterrows():
        if pd.isna(row['Original Country']):
            logger.error(f"{COLOR_RED}Original country is NaN for {row['Country']}, {row['Value']}, {row['Rule-of-Thumb']}, {row['Background']}{COLOR_NORMAL}")
            pdb.set_trace()
        else:
            original_row = story_df[(story_df['Country'] == row['Original Country']) & (story_df['Value'] == row['Original Value']) & (story_df['Rule-of-Thumb'] == row['Original Rule-of-Thumb']) & (story_df['Background'] == row['Original Background'])]
            if original_row.empty:
                logger.error(f"{COLOR_RED}Original row not found for {row['Country']}, {row['Value']}, {row['Rule-of-Thumb']}, {row['Background']}{COLOR_NORMAL}")
                pdb.set_trace()
            else:
                original_row = original_row.iloc[0]
                story_df.loc[ind, 'Original Response'] = original_row['Response']
                story_df.loc[ind, 'Original Resp_Explanation'] = original_row['Resp_Explanation']
    # Orig responses: cache the responses where the country is same as that of the original
    original_story_df = story_df[story_df['Country'] == story_df['Original Country']]
    # delete the rows where the country is the same as the original country
    story_df = story_df[story_df['Country'] != story_df['Original Country']]
    return story_df, original_story_df

def extract_response_explanation(answerstr: str) -> List[str]:
    try:
        response = answerstr.split('# Explanation:')[0].strip().strip('.')
    except:
        pdb.set_trace()
        response = '<<No Response provided>>'
    try:
        explanation = answerstr.split('# Explanation:')[1].strip()
    except:
        explanation = '<<No Explanation provided>>'
    return response, explanation

@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def main(cfg: DictConfig):
    # ask user to check config values
    print(f"{COLOR_RED} Please verify config values {COLOR_BLUE}")
    print(OmegaConf.to_yaml(cfg))
    user_input = input(f"{COLOR_GREEN}Continue? [y/n] [N] {COLOR_NORMAL}")
    if user_input.strip().lower() != 'y':
        print(f"{COLOR_RED}Exiting...{COLOR_NORMAL}")
        logger.debug(f"User chose {user_input}")
        exit()
    
    # read in the openai api key
    if 'gpt' in cfg.story_inference.model_name: 
        with open(f"{cfg.prompts.openai_key}", 'r') as f:
            OPENAI_API_KEY = f.read().strip()
        modelInferencer = OpenAIInferencer(openai_api_key=OPENAI_API_KEY, name=cfg.story_inference.model_name, log_batch=10)

    #TODO: Add other models
    story_df = pd.read_csv(cfg.story_inference.path_to_story_file,encoding='utf-8-sig')
    story_df = sample_iwslt_bin(story_df=story_df)
    if cfg.story_inference.do_test:
        story_df = story_df.sample(10)

    story_df, original_story_df = postprocess_dataframe(cfg=cfg, story_df=story_df)
    # get the responses for the sampled dataframe
    for condition, attrs in map_conditioning_to_attr.items():
        if not cfg.story_inference[condition]:
            continue
        
        file = attrs['file']+f'_{cfg.story_inference.model_name}'
        if cfg.story_inference.do_test:
            file += '_test'
        file += '.csv'
        
        prompts = attrs['prompt'](X=story_df)
        condition_responses = modelInferencer(prompts,
                                                max_tokens=cfg.story_inference.max_tokens, 
                                                temperature=cfg.story_inference.temperature)
        
        condition_resp, condition_expl = list(zip(*[extract_response_explanation(cr) for cr in condition_responses]))
        story_df['Response'] = condition_resp
        story_df['Resp_Explanation'] = condition_expl
        story_df, _ = postprocess_dataframe(cfg=cfg, story_df=story_df)
        story_df.to_csv(os.path.join(cfg.prompts.output_dir, file), index=False)

if __name__ == '__main__':
    main()










    


    


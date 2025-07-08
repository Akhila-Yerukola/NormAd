import hydra
from omegaconf import OmegaConf, DictConfig
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict
from collections import defaultdict
from src.analysis.utils import spider_plot
import pdb
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 300

COLOR_RED = '\033[91m'
COLOR_GREEN = '\033[92m'
COLOR_BLUE = '\033[94m'
COLOR_NORMAL = '\033[0m'


# TODO. the inglehart welzel cultural map is too biased. we should bin these differently.
map_iw_bin_to_country = {
    'African-Islamic': ['afghanistan', 'bangladesh','egypt','ethiopia','iran','iraq', 'kenya', 'lebanon', 'mauritius', 'pakistan','palestinian_territories','saudi_arabia', 'somalia', 'south_africa','south_sudan','sudan','syria', 'türkiye','zimbabwe'], # mauritius 
    'Orthodox Europe': ['bosnia_and_herzegovina','greece','north_macedonia','romania','russia', 'serbia','ukraine'],
    'Confucian': ['china','hong_kong','japan','south_korea','taiwan'],
    'Catholic Europe': ['austria','croatia','hungary','italy','poland','portugal','spain'],
    'Protestant Europe': ['france','germany','netherlands','sweden', 'malta'], # malta
    'English Speaking': ['australia','canada','ireland','new_zealand','united_kingdom','united_states_of_america'],
    'Latin America': ['argentina','brazil','chile','colombia','mexico','peru','venezuela'], 
    'West and South Asia': ['cambodia','cyprus','fiji','india','indonesia','israel','laos','malaysia','myanmar','nepal','papua_new_guinea','samoa','singapore','sri_lanka','thailand', 'timor-leste', 'tonga', 'vietnam','philippines'] # philippines, papua new guinea, samoa, timor-leste, tonga
}

map_country_to_iw_bin = {}
for iwbin,countries in map_iw_bin_to_country.items():
    for country in countries:
        map_country_to_iw_bin[country] = iwbin

save_iwbin_country_list = pd.DataFrame({'country': map_country_to_iw_bin.keys(), 'iw_bin': map_country_to_iw_bin.values()})
save_iwbin_country_list.to_csv('./src/analysis/country_iwlist.csv',encoding='utf-8-sig', index=False)

map_filename_to_config  = {
    # Will be populated in main
}

def get_dataframe(cfg: DictConfig, file_prefix: str) -> pd.DataFrame:
    model_name = cfg.story_inference.model_name.replace('/','_').replace('-','_')
    file_name = os.path.join(cfg.story_inference.save_path, 
                                f'{file_prefix}_{model_name}.csv')
    story_df = pd.read_csv(file_name, encoding='utf-8-sig')
    
    responses = story_df['Response'].apply(lambda x: x.split()[0])
    # strip punctuation, :,;,',. etc
    responses = responses.apply(lambda x: x.strip().strip('.').strip(',').strip(';').strip(':').strip('\''))
    unique_vals = list(responses.unique())

    unique_vals = set(unique_vals)
    # Assert that the unique values are a subset of the expected values ['No', 'Yes', 'Neutral', 'Unsure']
    assert unique_vals.issubset(set(['No', 'Yes', 'Neutral', 'Unsure'])), f"{COLOR_RED}Your file '{file_name}' has malformed responses. Unique responses are {unique_vals}. Check your prompt or postprocessing{COLOR_NORMAL}"
    story_df['Response'] = responses
    return story_df 

def plot_results(cfg: DictConfig):
    # use the config values to change this
    model_name = cfg.story_inference.model_name.replace('/','_').replace('-','_')
    map_filename_to_config = {
        "etiquette_non_conditioned": cfg.story_inference.no_condition,
        "etiquette_country_conditioned": cfg.story_inference.country_condition,
        "etiquette_value_conditioned": cfg.story_inference.value_condition,
        "etiquette_rot_conditioned": cfg.story_inference.rot_condition,
        "etiquette_bgd_conditioned": cfg.story_inference.bgd_condition,
        "etiquette_full_conditioned": cfg.story_inference.full_condition
    }

    map_filename_to_label = {
        "etiquette_non_conditioned": "completely unconditioned story",
        "etiquette_country_conditioned": "country conditioned story",
        "etiquette_value_conditioned": "value conditioned story",
        "etiquette_rot_conditioned": "rot conditioned story",
        "etiquette_bgd_conditioned": "background conditioned story",
        "etiquette_full_conditioned": "bgd+value+rot conditioned story"
    }

    # Plot model-level alignment scores
    def check_equal(response: pd.Series, gold: pd.Series) -> pd.Series:
        response = response.apply(lambda x: x.lower().strip().strip('.'))
        gold = gold.apply(lambda x: x.lower().strip().strip('.'))
        return response == gold

    fraction_eq = []
    for file_prefix, boolvalue in map_filename_to_config.items():
        if boolvalue:
            story_df = get_dataframe(cfg, file_prefix)
            # count No
            num_eq = len(story_df[check_equal(story_df['Response'], story_df['Gold Label'])])
            fraction_eq.append(num_eq/len(story_df))

            
    fig = plt.figure(figsize=(8,8))
    rects = plt.bar([map_filename_to_label[filename] for filename,config in map_filename_to_config.items() if config] , fraction_eq, width=0.5)
    plt.bar_label(rects, fmt="{:.2f}", padding=2)         
    plt.xticks(rotation=45)
    plt.title(f"Alignment of {model_name} across different conditionings")
    plt.subplots_adjust(bottom=0.30)
    plt.savefig(f'plots/{model_name}.png')
    plt.clf()   

    # Plot alignment scores per subaxis

    subaxes_counts = defaultdict(list)
    offset = 0
    for file_prefix, boolvalue in map_filename_to_config.items():
        if boolvalue:
            story_df = get_dataframe(cfg, file_prefix)
            # plot along axis 
            subaxes = []
            viable_group_names = {'basic_etiquette': 'basic_etiquette', 
                                  'visiting': 'visiting', 
                                  'eating': 'eating', 
                                  'gifts': 'gifts',
                                  'gift-giving': 'gifts',
                                  'gift_giving': 'gifts',}
            
            for name, group in story_df.groupby('Subaxis'):
                # consider plotting only for the 4 most frequent subaxes (the rest are country-specific)
                if name in viable_group_names.keys():
                    num_eq_group = len(group[check_equal(group['Response'], group['Gold Label'])])
                    fraction_eq_group  = num_eq_group/len(group)
                    subaxes_counts[viable_group_names[name]].append(fraction_eq_group)
             
    # transpose subaxes_counts
    subaxes = subaxes_counts.keys()
    file_groups = defaultdict(list)
    for k,v in subaxes_counts.items():
        # get true map_filename_to_config keys
        true_filename_config_keys = [filename for filename,config in map_filename_to_config.items() if config]
        for value, file_prefix in zip(v,true_filename_config_keys):
            file_groups[file_prefix].append(value)
    
    x = np.arange(len(subaxes))  # the label locations
    width = 0.16  # the width of the bars
    multiplier = 0

    plt.rcParams['font.size'] = 6
    fig = plt.figure(figsize=(8,6))
    for attribute, measurement in file_groups.items():
        offset = width * multiplier
        rects = plt.bar(x + offset, measurement, width, label=map_filename_to_label[attribute])
        plt.bar_label(rects, fmt="{:.2f}", padding=2)
        multiplier += 1

    plt.rcParams['font.size'] = 10
    plt.legend(loc='lower center', bbox_to_anchor=[0.5, -0.5])
    plt.xticks(x+width, subaxes, rotation=45)
    plt.subplots_adjust(bottom=0.30)
    plt.title(f"Alignment of {model_name} across subaxes")
    plt.savefig(f'plots/{model_name}_subaxes.png')
    plt.clf()   

    # plot cultural alignment per iw_bin
    
    iw_bin_counts = defaultdict(list)
    for file_prefix, boolvalue in map_filename_to_config.items():
        if boolvalue:
            story_df = get_dataframe(cfg, file_prefix)
            # plot along country bin
            iw_bins, nos = [], []
            def country_tr(x: Dict) -> Dict:
                x['Country'] = map_country_to_iw_bin[x['Country']]
                return x

            story_iw_bin_df = story_df.apply(country_tr, axis=1)

            for name, group in story_iw_bin_df.groupby('Country'):
                iw_bin = name
                num_eq_group =  len(group[check_equal(group['Response'], group['Gold Label'])])
                fraction_eq_group  = num_eq_group/len(group)
                iw_bin_counts[iw_bin].append(fraction_eq_group)        
            
    iw_bins = iw_bin_counts.keys()
    file_groups = defaultdict(list)
    for k,v in iw_bin_counts.items():
        # get true map_filename_to_config keys
        true_filename_config_keys = [filename for filename,config in map_filename_to_config.items() if config]
        for value, file_prefix in zip(v,true_filename_config_keys):
            file_groups[file_prefix].append(value)

    x = np.arange(len(iw_bins))
    width = 0.16
    multiplier = 0

    spider_plot({'categories': list(iw_bins), 'values': list(file_groups.values())}, 
                [map_filename_to_label[attribute] for attribute in file_groups.keys()], 
                f"Alignment of {model_name} across inglehart welzel bins", 
                f'plots/{model_name}_iwbin.png')
    # fig = plt.figure(figsize=(10,6))
    # plt.rcParams['font.size'] = 4.5
    # for attribute, measurement in file_groups.items():
    #     offset = width * multiplier
    #     rects = plt.bar(x + offset, measurement, width, label=map_filename_to_label[attribute])
    #     plt.bar_label(rects, fmt="{:.2f}", padding=2)
    #     multiplier += 1

    # plt.rcParams['font.size'] = 10
    # plt.legend(loc='lower center', bbox_to_anchor=[0.5,-0.5])
    # plt.xticks(x+width, iw_bins, rotation=45)
    # plt.subplots_adjust(bottom=0.3)
    # plt.title(f"Alignment of {cfg.story_inference.model_name} across inglehart welzel bins")
    # plt.savefig(f'plots/{cfg.story_inference.model_name}_iwbin.png')
    # plt.clf()       


@hydra.main(version_base=None, config_name='config', config_path='../../conf')
def main(cfg: DictConfig) -> None:
    # ask user to check config values
    print(f"{COLOR_RED} Please verify config values {COLOR_BLUE}")
    print(OmegaConf.to_yaml(cfg))
    user_input = input(f"{COLOR_GREEN}Continue? [y/n] [N] {COLOR_NORMAL}")
    if user_input.strip().lower() != 'y':
        print(f"{COLOR_RED}Exiting...{COLOR_NORMAL}")
        exit()

    plot_results(cfg)


 
            



if __name__ == '__main__':
    main()
    
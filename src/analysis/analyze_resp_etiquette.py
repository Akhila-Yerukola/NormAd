import hydra
from omegaconf import OmegaConf, DictConfig
from collections import OrderedDict
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict
from collections import defaultdict
# from src.analysis.utils import spider_plot
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

group_mapping = OrderedDict({
    'basic_etiquette': 'basic_etiquette', 
    'manners_in_vietnam': 'basic_etiquette',
    'māori_etiquette': 'basic_etiquette',
    'cleanliness': 'basic_etiquette',
    'direct_manners': 'basic_etiquette',
    'tipping': 'basic_etiquette', 
    '‘taarof’_(politeness_and_mutual_respect)': 'basic_etiquette',
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
    })

map_country_to_iw_bin = {}
for iwbin,countries in map_iw_bin_to_country.items():
    for country in countries:
        map_country_to_iw_bin[country] = iwbin

# save the country to iw_bin mapping
save_iwbin_country_list = pd.DataFrame({'country': map_country_to_iw_bin.keys(), 'iw_bin': map_country_to_iw_bin.values()})
save_iwbin_country_list.to_csv('country_iwlist.csv',encoding='utf-8-sig', index=False)

# save the group mapping
save_group_mapping = pd.DataFrame({'subaxis': group_mapping.keys(), 'axis': group_mapping.values()})
save_group_mapping.to_csv('group_mapping.csv',encoding='utf-8-sig', index=False)

# map_filename_to_config  = {
#     # Will be populated in main
# }

def get_dataframe(file_path, model_name, file_prefix: str) -> pd.DataFrame:
    #model_name = cfg.story_inference.model_name.replace('/','_') # .replace('-','_')
    file_name = os.path.join(file_path, 
                                f'{file_prefix}_{model_name}.csv')
    try:
        story_df = pd.read_csv(file_name, encoding='utf-8-sig')
    except:
        model_name = model_name.replace('-','_')
        file_name = os.path.join(file_path,
                                f'{file_prefix}_{model_name}.csv')
        story_df = pd.read_csv(file_name, encoding='utf-8-sig')
    # assert that only Yes and No exist in the dataframe
        
    # rename prediction_label_temp0.0 to Response
    if 'prediction_label_temp0.0' in story_df.columns:
        col = 'prediction_label_temp0.0'
    else:
        col = 'prediction_label_temp0.001'
    story_df.rename(columns={col: 'Response'}, inplace=True)
    story_df['Gold Label'] = story_df['Gold Label'].apply(lambda x: 'neither' if x.lower() == 'neutral' else x)
    story_df['Response'] = story_df['Response'].apply(lambda x: 'neither' if x.lower() == 'neutral' else x)
    # do some additional preprocessing
    story_df['Response'] = story_df['Response'].apply(lambda x: 'yes' if 'yes' in x.lower() else x)
    story_df['Response'] = story_df['Response'].apply(lambda x: 'no' if 'no' in x.lower() else x)
    story_df['Response'] = story_df['Response'].apply(lambda x: 'neither' if 'neither' in x.lower() else x)
    
    responses = story_df['Response'].apply(lambda x: x.split()[0])
    # responses = responses.apply(lambda x: 'yes' if x == '1)' else x)
    # responses = responses.apply(lambda x: 'no' if x == '2)' else x)
    # responses = responses.apply(lambda x: 'neither' if x == '3)' else x)
    
    # strip punctuation, :,;,',. etc
    responses = responses.apply(lambda x: x.strip().strip('.').strip(',').strip(';').strip(':').strip('\'').lower())
    unique_vals = list(responses.unique())

    unique_vals = set(unique_vals)
    # remove all stories that don't have a response in ['yes', 'no', 'neither']
    # story_df = story_df[story_df['Response'].apply(lambda x: x in ['yes', 'no', 'neither'])]
    assert unique_vals.issubset(set(['no', 'yes', 'neither'])), f"{COLOR_RED}Your file '{file_name}' has malformed responses. Unique responses are {unique_vals}. Check your prompt or postprocessing{COLOR_NORMAL}"
    story_df['Response'] = responses

    # remove stories that have gold label == 'Neutral'
    # story_df = story_df[story_df['Gold Label'] != 'neutral']
    return story_df 

def plot_results(file_path, model_name: str):
    # use the config values to change this
    map_filename_to_config = OrderedDict({
        "etiquette_none_conditioned": True,
        "etiquette_country_conditioned": True,
        "etiquette_value_conditioned": True,
        "etiquette_rot_conditioned": True,
    })

    map_filename_to_label = OrderedDict({
        "etiquette_none_conditioned": "baseline reference performance",
        "etiquette_country_conditioned": "country conditioned story",
        "etiquette_value_conditioned": "value conditioned story",
        "etiquette_rot_conditioned": "rot conditioned story",
    })

    # Plot model-level Adaptability scores
    def check_equal(response: pd.Series, gold: pd.Series) -> pd.Series:
        response = response.apply(lambda x: x.lower().strip().strip('.'))
        gold = gold.apply(lambda x: x.lower().strip().strip('.'))
        equal_vals = []
        for r, g in zip(response, gold):
            if g in ['yes', 'no', 'neither']:
                equal_vals.append(r==g)
            else:
                pdb.set_trace()
        
        equal_vals = pd.Series(equal_vals)
        # align index
        equal_vals.index = response.index
        assert all(equal_vals.index == gold.index)
        return equal_vals

    def check_precision(response: pd.Series, gold: pd.Series) -> pd.Series:
        response = response.apply(lambda x: x.lower().strip().strip('.'))
        gold = gold.apply(lambda x: x.lower().strip().strip('.'))
        precision_vals = []
        normalized_responses = response
        normalized_gold = gold
        unique_labels = ['yes', 'no', 'neither']
        for label in unique_labels:
            tp = sum((normalized_responses == label) & (normalized_gold == label))
            fp = sum((normalized_responses == label) & (normalized_gold != label))
            if tp+fp == 0:
                precision = 0
            else:
                precision = tp/(tp+fp)
            precision_vals.append(precision)
        return pd.Series(precision_vals, index=unique_labels)
    
    def check_recall(response: pd.Series, gold: pd.Series) -> pd.Series:
        response = response.apply(lambda x: x.lower().strip().strip('.'))
        gold = gold.apply(lambda x: x.lower().strip().strip('.'))
        recall_vals = []
        normalized_responses = response
        normalized_gold = gold
        unique_labels = ['yes','no', 'neither'] # normalized_gold.unique()
        for label in unique_labels:
            tp = sum((normalized_responses == label) & (normalized_gold == label))
            fn = sum((normalized_responses != label) & (normalized_gold == label))
            if tp+fn == 0:
                recall = 0
            else:
                recall = tp/(tp+fn)
            recall_vals.append(recall)
        return pd.Series(recall_vals, index=unique_labels)
    
    def check_f1(response: pd.Series, gold: pd.Series) -> pd.Series:
        precision = check_precision(response, gold)
        recall = check_recall(response, gold)

        f1 = 2*(precision*recall)/(precision+recall+1e-6)
        return f1

    def plot_acc():
        fraction_eq = []
        stddev_eq = []
        dfs = {}
        for file_prefix, boolvalue in map_filename_to_config.items():
            if boolvalue:
                story_df = get_dataframe(file_path, model_name, file_prefix)
                if 'none_conditioned' in file_prefix:
                    continue
                # count No
                num_eq = len(story_df[check_equal(story_df['Response'], story_df['Gold Label'])])
                fraction_eq.append(num_eq/len(story_df))
                sorted_story_df = story_df.sort_values(by=['Axis','Subaxis'])
                num_samples = 3
                sdfs = [sorted_story_df.iloc[i::num_samples].reset_index(drop=True) for i in range(num_samples)]
                frac_eqs = [len(sdf[check_equal(sdf['Response'], sdf['Gold Label'])])/len(sdf) for sdf in sdfs]
                stddev_eq.append(np.std(frac_eqs))


        # fig = plt.figure(figsize=(8,8))
        fig, ax = plt.subplots()
        fig.set_size_inches(8, 8)
        rects = ax.bar([map_filename_to_label[filename] for filename,config in map_filename_to_config.items() if config and 'none_conditioned' not in filename] , fraction_eq, width=0.5)
        # add error bars
        ax.errorbar([map_filename_to_label[filename] for filename,config in map_filename_to_config.items() if config and 'none_conditioned' not in filename], fraction_eq, yerr=stddev_eq, fmt='-', color='black', ecolor='black', elinewidth=1, capsize=5)
        ax.bar_label(rects, fmt="{:.2f}", padding=2)   
        ax.tick_params(axis='x', rotation=45)
        ax.set_ylim(0,1)
        ax.set_title(f"Adaptability of {model_name} across different conditionings \n(accuracy [response == gold label], std dev across 3 samples)")
        plt.subplots_adjust(bottom=0.30)
        plt.savefig(f'{file_path}/{model_name}_overall.png')
        plt.clf()   

    # Plot adaptability scores per subaxis
    def plot_subaxes():
        subaxes_counts = defaultdict(list)
        subaxes_stddevs = defaultdict(list)
        offset = 0
        for file_prefix, boolvalue in map_filename_to_config.items():
            if boolvalue:
                story_df = get_dataframe(file_path, model_name, file_prefix)
                if 'none_conditioned' in file_prefix:
                    story_df = story_df[story_df['Gold Label'] != 'neutral']
                # plot along axis 
                subaxes = []
                for name, group in story_df.groupby('Subaxis'):
                    # consider plotting only for the 4 most frequent subaxes (the rest are country-specific)
                    if name in group_mapping.keys():
                        num_eq_group = len(group[check_equal(group['Response'], group['Gold Label'])])
                        fraction_eq_group  = num_eq_group/len(group)
                        subaxes_counts[group_mapping[name]].append(fraction_eq_group)
                        # get stddevs 
                        num_samples = 3
                        # sort group by axis
                        sorted_group = group.sort_values(by=['Axis'])
                        sdfs = [sorted_group.iloc[i::num_samples].reset_index(drop=True) for i in range(num_samples)]
                        frac_eqs = [len(sdf[check_equal(sdf['Response'], sdf['Gold Label'])])/len(sdf) for sdf in sdfs]
                        subaxes_stddevs[group_mapping[name]].append(np.std(frac_eqs))
                    else:
                        # shouldn't be happening as we account for everything, hit debug
                        pdb.set_trace()

             
        # transpose subaxes_counts
        subaxes = subaxes_counts.keys()
        file_groups = defaultdict(list)
        file_groups_stddevs = defaultdict(list)
        for (kc,vc), (ks, vs) in zip(subaxes_counts.items(), subaxes_stddevs.items()):
            # get true map_filename_to_config keys
            true_filename_config_keys = [filename for filename,config in map_filename_to_config.items() if config]
            for value, file_prefix in zip(vc,true_filename_config_keys):
                file_groups[file_prefix].append(value)

            for value, file_prefix in zip(vs,true_filename_config_keys):
                file_groups_stddevs[file_prefix].append(value)
        
        x = np.arange(len(subaxes))  # the label locations
        width = 0.16  # the width of the bars
        multiplier = 0

        plt.rcParams['font.size'] = 6
        # fig = plt.figure(figsize=(8,6))
        fig, ax = plt.subplots()
        fig.set_size_inches(8, 6)
        for attribute, measurement in file_groups.items():
            offset = width * multiplier
            rects = plt.bar(x + offset, measurement, width, label=map_filename_to_label[attribute])
            plt.errorbar(x + offset, measurement, yerr=file_groups_stddevs[attribute], fmt='.', color='black', ecolor='gray', elinewidth=1, capsize=5)
            plt.bar_label(rects, fmt="{:.2f}", padding=2)
            multiplier += 1

        plt.rcParams['font.size'] = 10
        ax.legend(loc='lower center', bbox_to_anchor=[0.5, -0.5])
        ax.set_xticks(x+width, subaxes, rotation=45)
        ax.set_ylim(0,1)
        plt.subplots_adjust(bottom=0.30)
        ax.set_title(f"Adaptability of {model_name} across subaxes \n(accuracy [response == gold label], std dev across 3 samples)")
        plt.savefig(f'{file_path}/{model_name}_subaxes.png')
        plt.clf()   

    # plot cultural Adaptability per iw_bin
    def plot_iw_bin():
        iw_bin_counts = defaultdict(list)
        iw_bin_stddevs = defaultdict(list)
        non_conditioned_accs = []
        for file_prefix, boolvalue in map_filename_to_config.items(): 
            if boolvalue:
                story_df = get_dataframe(file_path, model_name, file_prefix)
                # if 'non_conditioned' in file_prefix:
                #     story_df = story_df[story_df['Gold Label'] != 'neutral']
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
                    # get stddevs
                    num_samples = 3
                    # sort group by axis, subaxis
                    sorted_group = group.sort_values(by=['Axis','Subaxis'])
                    sdfs = [sorted_group.iloc[i::num_samples].reset_index(drop=True) for i in range(num_samples)]
                    frac_eqs = [len(sdf[check_equal(sdf['Response'], sdf['Gold Label'])])/len(sdf) for sdf in sdfs]
                    iw_bin_stddevs[iw_bin].append(np.std(frac_eqs))

                
        iw_bins = iw_bin_counts.keys()
        file_groups = defaultdict(list)
        file_groups_stddevs = defaultdict(list)
        for (kc,vc), (ks, vs) in zip(iw_bin_counts.items(), iw_bin_stddevs.items()):
            # get true map_filename_to_config keys
            true_filename_config_keys = [filename for filename,config in map_filename_to_config.items() if config]
            for value, file_prefix in zip(vc,true_filename_config_keys):
                file_groups[file_prefix].append(value)
            
            for value, file_prefix in zip(vs,true_filename_config_keys):
                file_groups_stddevs[file_prefix].append(value)

        x = np.arange(len(iw_bins))
        width = 0.24
        multiplier = 0

        plt.rcParams['font.size'] = 6
        # fig = plt.figure(figsize=(8,6))
        fig, ax = plt.subplots()
        fig.set_size_inches(8, 6)
        for attribute, measurement in file_groups.items():
            if 'none_conditioned' in attribute:
                continue
            offset = width * multiplier
            rects = plt.bar(x + offset, measurement, width, label=map_filename_to_label[attribute])
            plt.errorbar(x + offset, measurement, yerr=file_groups_stddevs[attribute], fmt='.', color='black', ecolor='gray', elinewidth=1, capsize=5)
            plt.bar_label(rects, fmt="{:.2f}", padding=2)
            multiplier += 1
        
        # if non_conditioned in the filegroups, plot it as a baseline
        if any(['none_conditioned' in value for value in file_groups.keys()]):
            attribute, measurement = list(file_groups.items())[0]
            offset = width * multiplier
            X = [[x-width/2, x+offset-width/2, None] for x in range(len(iw_bins))]
            X = [item for sublist in X for item in sublist]
            Y = [[y, y, None] for y in measurement]
            Y = [item for sublist in Y for item in sublist]
            # plot a dotted line across the bars
            plt.plot(X,Y, linestyle='dotted', color='black', label=map_filename_to_label[attribute])


        plt.rcParams['font.size'] = 10
        ax.legend(loc='lower center', bbox_to_anchor=[0.5, -0.5])
        ax.set_xticks(x+width, iw_bins, rotation=45)
        ax.set_ylim(0,1)
        plt.subplots_adjust(bottom=0.30)
        ax.set_title(f"Adaptability of {model_name} across Inglehart-Welzel bins \n(accuracy [response == gold label], std dev across 3 samples)")
        plt.savefig(f'{file_path}/{model_name}_iwbin.png')
        plt.clf()   

        # spider_plot({'categories': list(iw_bins), 'values': list(file_groups.values()), 'stddev': list(file_groups_stddevs.values())},
        #             [map_filename_to_label[attribute] for attribute in file_groups.keys()], 
        #             f"Adaptability of {model_name} across Inglehart-Welzel bins \n(accuracy [response == gold label], std dev across 3 samples", 
        #             f'plots/{model_name}_iwbin.png')
        # plt.clf()
    
    # get accuracy per label per conditioning
    def plot_labels():
        label_counts = defaultdict(list)
        label_stddevs = defaultdict(list)
        for file_prefix, boolvalue in map_filename_to_config.items():
            if boolvalue:
                story_df = get_dataframe(file_path, model_name, file_prefix)
                # plot along country bin
                labels = ['yes', 'no', 'neither']
                for label in labels:
                    group = story_df[story_df['Gold Label'] == label]
                    num_eq_group =  len(group[check_equal(group['Response'], group['Gold Label'])])
                    fraction_eq_group  = num_eq_group/len(group)
                    label_counts[label].append(fraction_eq_group)
                    # get stddevs
                    num_samples = 3
                    # sort group by axis, subaxis
                    sorted_group = group.sort_values(by=['Axis','Subaxis'])
                    sdfs = [sorted_group.iloc[i::num_samples].reset_index(drop=True) for i in range(num_samples)]
                    frac_eqs = [len(sdf[check_equal(sdf['Response'], sdf['Gold Label'])])/len(sdf) for sdf in sdfs]
                    label_stddevs[label].append(np.std(frac_eqs))

        file_groups = defaultdict(list)
        file_groups_stddevs = defaultdict(list)
        for (kc,vc), (ks, vs) in zip(label_counts.items(), label_stddevs.items()):
            # get true map_filename_to_config keys
            true_filename_config_keys = [filename for filename,config in map_filename_to_config.items() if config]
            for value, file_prefix in zip(vc,true_filename_config_keys):
                file_groups[file_prefix].append(value)
            
            for value, file_prefix in zip(vs,true_filename_config_keys):
                file_groups_stddevs[file_prefix].append(value)

        # spider_plot({'categories': list(labels), 'values': list(file_groups.values()), 'stddev': list(file_groups_stddevs.values())},
        #             [map_filename_to_label[attribute] for attribute in file_groups.keys()], 
        #             f"Adaptability of {model_name} across labels \n(accuracy [response == gold label], std dev across 3 samples)", 
        #             f'plots/{model_name}_labels.png')
        
        x = np.arange(len(labels))  # the label locations
        width = 0.16  # the width of the bars
        multiplier = 0

        plt.rcParams['font.size'] = 6
        # fig = plt.figure(figsize=(8,6))
        fig, ax = plt.subplots()
        fig.set_size_inches(8, 6)
        for attribute, measurement in file_groups.items():
            if 'none_conditioned' in attribute:
                continue
            offset = width * multiplier
            rects = plt.bar(x + offset, measurement, width, label=map_filename_to_label[attribute])
            plt.errorbar(x + offset, measurement, yerr=file_groups_stddevs[attribute], fmt='.', color='black', ecolor='gray', elinewidth=1, capsize=5)
            plt.bar_label(rects, fmt="{:.2f}", padding=2)
            multiplier += 1
        
        # if non_conditioned in the filegroups, plot it as a baseline
        if any(['none_conditioned' in value for value in file_groups.keys()]):
            attribute, measurement = list(file_groups.items())[0]
            offset = width * multiplier
            X = [[x-width/2, x+offset-width/2, None] for x in range(len(labels))]
            X = [item for sublist in X for item in sublist]
            Y = [[y, y, None] for y in measurement]
            Y = [item for sublist in Y for item in sublist]
            # plot a dotted line across the bars
            plt.plot(X,Y, linestyle='dotted', color='black', label=map_filename_to_label[attribute])
        
        plt.rcParams['font.size'] = 10
        ax.legend(loc='lower center', bbox_to_anchor=[0.5, -0.5])
        ax.set_xticks(x+width, labels, rotation=45)
        ax.set_ylim(0,1)
        plt.subplots_adjust(bottom=0.30)
        ax.set_title(f"Adaptability of {model_name} across labels \n(accuracy [response == gold label],\nstd dev across 3 samples)")
        plt.savefig(f'{file_path}/{model_name}_labels.png')
        plt.clf()   
        
        plt.clf()

    # get precision, recall and f-score per conditioning
    def get_pr_rc_f1():
        precision_counts = defaultdict(list)
        recall_counts = defaultdict(list)
        f1_counts = defaultdict(list)
        for file_prefix, boolvalue in map_filename_to_config.items():
            if boolvalue:
                story_df = get_dataframe(file_path, model_name, file_prefix)
                # plot along country bin
                labels = ['yes', 'no', 'neither'] #if 'non_conditioned' not in file_prefix else ['yes', 'no',]
                precision = check_precision(story_df['Response'], story_df['Gold Label'])
                recall = check_recall(story_df['Response'], story_df['Gold Label'])
                f1 = check_f1(story_df['Response'], story_df['Gold Label'])
                for label in labels:
                    precision_counts[label].append(precision[label])
                    recall_counts[label].append(recall[label])
                    f1_counts[label].append(f1[label])
        # save as a single, multi-indexed pretty-printed dataframe that looks like this:
        columns = ['precision', 'recall', 'f1']
        rows = ['yes', 'no', 'neither']
        row_groups = [map_filename_to_label[filename] for filename,config in map_filename_to_config.items() if config]
        multi_index = pd.MultiIndex.from_product([row_groups, rows], names=['conditioning', 'label'])
        precision_recall_f1_df = pd.DataFrame(index=multi_index, columns=columns)
        for (kc,vc), (ks, vs), (kf, vf) in zip(precision_counts.items(), recall_counts.items(), f1_counts.items()):
            for value, file_prefix in zip(vc, row_groups):
                precision_recall_f1_df.loc[(file_prefix, kc), 'precision'] = value
            for value, file_prefix in zip(vs, row_groups):
                precision_recall_f1_df.loc[(file_prefix, ks), 'recall'] = value
            for value, file_prefix in zip(vf, row_groups):
                precision_recall_f1_df.loc[(file_prefix, kf), 'f1'] = value
        # set fp precision to 3 decimal places
        precision_recall_f1_df = precision_recall_f1_df.map(lambda x: round(x, 3) if isinstance(x, float) else x)
        precision_recall_f1_df.to_csv(f'{file_path}/{model_name}_precision_recall_f1.tsv', encoding='utf-8-sig', float_format='%.3f', sep='\t')
        # get latex
        precision_recall_f1_df.to_latex(f'{file_path}/{model_name}_precision_recall_f1.tex', encoding='utf-8-sig', float_format='%.3f', 
                                        caption=f'Precision, Recall and F1 scores for {model_name} across different conditionings and labels', column_format='l|l|rrr')
        
    plot_acc()
    plot_subaxes()
    plot_iw_bin()
    plot_labels()
    get_pr_rc_f1()
    mpl.pyplot.close('all')


def main() -> None:
    # ask user to check config values
    # print(f"{COLOR_RED} Please verify config values {COLOR_BLUE}")
    # print(OmegaConf.to_yaml(cfg))
    # user_input = input(f"{COLOR_GREEN}Continue? [y/n] [N] {COLOR_NORMAL}")
    # if user_input.strip().lower() != 'y':
    #     print(f"{COLOR_RED}Exiting...{COLOR_NORMAL}")
    #     exit()
    for (dirpath, dirnames, filenames) in os.walk('./model_outputs/all_model_outputs', topdown=True):
        # for filename in filenames:
        #     if filename.endswith('.csv'):
        #         pdb.set_trace()
        #         model_name = filename.split('_')[0]
        for dirname in dirnames:
            # print(os.path.join(dirpath, dirname))
            # check if the directory has any files
            for (dirpath2, dirnames2, filenames2) in os.walk(os.path.join(dirpath, dirname)):
                # for filename2 in filenames2:
                #     if filename2.endswith('.csv'):
                if 'archangel' in dirname:
                    model_name = dirname.split('_')[1:]
                    model_name = '_'.join(model_name)
                elif 'Llama-2' in dirname:
                    model_name = dirname.replace('Llama-2', 'llama2')
                    model_name = model_name.lower()
                    # remove the '-hf' suffix
                    model_name = model_name[:-3]
                elif 'Mistral' in dirname:
                    model_name = 'mistral-chat'
                else:
                    model_name = dirname
                print(model_name)
                plot_results(dirpath2, model_name)
                # plot_results(dirname, model_name)
                        # print(f"{COLOR_GREEN}Plotting results for {model_name}{COLOR_NORMAL}")
                        # print(os.path.join(dirpath, filename))
                
    # plot_results()


 
            



if __name__ == '__main__':
    main()
    

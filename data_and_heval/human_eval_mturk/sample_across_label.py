import argparse
import pandas as pd
import os
import numpy as np
from itertools import permutations

def sample_four(df):
    # first, group by subaxis
    # then sample one label from each subaxis (yes, no, neutral)
    # the last sample can be any label at random
    # return the four samples
    if len(df) <= 4:
        return df
    
    samples = pd.DataFrame()
    subaxes = df['Subaxis'].unique()

    # groupby subaxes and get value counts of each label in subaxes
    counts = df.groupby('Subaxis')['Gold Label'].value_counts()
    
    three_sampled = False
    two_sampled = False
    if len(subaxes) >= 3:
        for permutation in permutations(subaxes, 3):
            # check if we can sample a 'yes' from the first subaxes 
            # 'no' from the second 
            # 'neutral' from the third
            # if not, try the next permutation
            if counts[permutation[0]].get('yes', 0) > 0 and counts[permutation[1]].get('no', 0) > 0 and counts[permutation[2]].get('neutral', 0) > 0:
                # sample them
                sample = df[df['Subaxis'] == permutation[0]]
                sample = sample[sample['Gold Label'] == 'yes'].sample(1)
                samples = pd.concat([sample, samples])

                sample = df[df['Subaxis'] == permutation[1]]
                sample = sample[sample['Gold Label'] == 'no'].sample(1)
                samples = pd.concat([sample, samples])

                sample = df[df['Subaxis'] == permutation[2]]
                sample = sample[sample['Gold Label'] == 'neutral'].sample(1)
                samples = pd.concat([sample, samples])

                # random sample the last one
                remaining_subaxis = [subaxis for subaxis in subaxes if subaxis not in permutation]
                if remaining_subaxis:
                    sample = df[df['Subaxis'] == np.random.choice(remaining_subaxis)].sample(1)
                    samples = pd.concat([sample, samples])
                
                else:
                    print(subaxes, permutation)
                    # get set difference of samples and df (to handle duplicates)
                    df_not = df[~df['ID'].isin(samples['ID'])]
                    sample = df_not.sample(1)
                    samples = pd.concat([sample, samples])

                three_sampled = True
                two_sampled = True
                print(len(samples))
                break
    
    if not three_sampled and len(subaxes) >= 2:
        print("twosampling")
        for permutation in permutations(subaxes, 2):
            label_perms = permutations(['yes', 'no', 'neutral'], 2)
            for label_perm in label_perms:
                if counts[permutation[0]].get(label_perm[0], 0) > 0 and counts[permutation[1]].get(label_perm, 0) > 0:
                    # sample them
                    sample = df[df['Subaxis'] == permutation[0]]
                    sample = sample[sample['Gold Label'] == 'yes'].sample(1)
                    samples = pd.concat([sample, samples])

                    sample = df[df['Subaxis'] == permutation[1]]
                    sample = sample[sample['Gold Label'] == 'no'].sample(1)
                    samples = pd.concat([sample, samples])

                    # sample the last two
                    remaining_subaxis = [subaxis for subaxis in subaxes if subaxis not in permutation]
                    if len(remaining_subaxis) == 2:
                        sample = df[df['Subaxis'] == remaining_subaxis[0]]
                        sample = sample.sample(1)
                        samples = pd.concat([sample, samples])

                        sample = df[df['Subaxis'] == remaining_subaxis[1]]
                        sample = sample.sample(1)
                        samples = pd.concat([sample, samples])
                    
                    elif len(remaining_subaxis) == 1:
                        sample = df[df['Subaxis'] == remaining_subaxis[0]]
                        try:
                            sample = sample.sample(2)
                            samples = pd.concat([sample, samples])
                        except:
                            # sample the last datapoint at random
                            sample = sample.sample(1)
                            samples = pd.concat([sample, samples])

                            # get set difference of samples and df (to handle duplicates)
                            df_not = df[~df['ID'].isin(samples['ID'])]
                            sample = df_not.sample(1)
                            samples = pd.concat([sample, samples])

                    two_sampled = True

                    break
    
    if not two_sampled and len(subaxes) >= 1:
        # for some reason just ONE label is present in all subaxes
        # just sample one datapoint from each subaxis
        sample_counts = 0
        for subaxis in subaxes:
            sample = df[df['Subaxis'] == subaxis].sample(1)
            samples = pd.concat([sample, samples])
            sample_counts += 1

        # somehow get 4 samples for each country
        remaining_samples = 4 - sample_counts
        if remaining_samples > 0:
            df_not = df[~df['ID'].isin(samples['ID'])]
            sample = df_not.sample(remaining_samples)
            samples = pd.concat([sample, samples])

    # sanity check
    if len(samples) != 4:
        raise ValueError(f"Could not sample 4 samples from the data len={len(samples)}")
    
    return samples


def generate_subset(file_path):
    source_df = pd.read_csv(os.path.join(file_path, "../datasets/normad_etiquette_final_data.csv"))
    
    group_names = {'basic_etiquette': 'basic_etiquette', 
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
                                  }
    source_df['Subaxis'] = source_df['Subaxis'].replace(group_names)
    
    final_subset = pd.DataFrame()
    
    # groupby country
    for country in source_df['Country'].unique():
        country_subset = source_df[source_df['Country'] == country]
        
        foursample = sample_four(country_subset)
        final_subset = pd.concat([final_subset, foursample])


    return final_subset

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default="./")
    parser.add_argument("--output_dir", type=str, default="human_eval_data")
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    results_df = generate_subset(args.input_dir)

    # results_df.to_csv(os.path.join(args.output_dir, "300_human_subset.csv"), index=False)
    
    # keep only some columns
    columns = ['ID', 'Country', 'Background', 'Value', 'Subaxis', 'Rule-of-Thumb', 'Story', 'Gold Label', 'Other Country', 'Other Background']
    results_df = results_df[columns]

    results_df['Background'] = results_df['Background'].str.split("\n").str[1:].str.join("\n").str.replace("\n", "</li>\n").str.replace("-", "<li>")
    
    # replace 'neutral' with 'irrelevant' in Gold Label
    results_df['Gold Label'] = results_df['Gold Label'].replace('neutral', 'irrelevant')

    # title case the country names and labels
    results_df['Country'] = results_df['Country'].str.replace("_", " ").str.title()
    results_df['Gold Label'] = results_df['Gold Label'].str.title()

    # rename columns
    results_df = results_df.rename(columns={'Gold Label': 'label'})
    results_df.columns = results_df.columns.str.lower().str.replace("-", "_")
    results_df.columns = results_df.columns.str.replace(" ", "_")

    # get stats
    print(results_df['country'].value_counts())
    print(results_df['label'].value_counts())
    print(results_df['subaxis'].value_counts())
    print(len(results_df))

    results_df.to_csv(os.path.join(args.output_dir, "300_human_subset_w_diverse_labels.csv"), index=False)
import argparse
import pandas as pd
import os

def generate_subset(file_path):
    source_df = pd.read_csv(os.path.join(file_path, "full_data_gpt4_3options.csv"))
    iw_df = pd.read_csv(os.path.join(file_path, "country_iwlist.csv"))
    combined_df = pd.merge(source_df, iw_df, left_on="Country", right_on="country", how="inner")
    
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
    combined_df['Subaxis'] = combined_df['Subaxis'].replace(group_names)
    
    final_subset = combined_df.groupby(['iw_bin', 'Gold Label', 'Subaxis']).apply(lambda x: x.sample(min(len(x), 5))).reset_index(drop=True)
    return final_subset

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default="data/")
    parser.add_argument("--output_dir", type=str, default="human_eval_data")
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    results_df = generate_subset(args.input_dir)
    results_df.to_csv(os.path.join(args.output_dir, "480_human_subset.csv"))
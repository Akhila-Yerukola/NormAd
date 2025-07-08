import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
import os
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
mpl.rcParams['figure.dpi'] = 300

file_prefixes = {
    "7b_sft": "output/ContextualAI/archangel_sft_llama7b/",
    "7b_sft_ppo": "output/ContextualAI/archangel_sft_ppo_llama7b/",
    "7b_sft_dpo": "output/ContextualAI/archangel_sft_dpo_llama7b/",
    "7b_sft_kto": "output/ContextualAI/archangel_sft_kto_llama7b/",
    "13b_sft": "output/ContextualAI/archangel_sft_llama13b/",
    "13b_sft_ppo": "output/ContextualAI/archangel_sft_ppo_llama13b/",
    "13b_sft_dpo": "output/ContextualAI/archangel_sft_dpo_llama13b/",
    "13b_sft_kto": "output/ContextualAI/archangel_sft_kto_llama13b/",
    "30b_sft": "output/ContextualAI/archangel_sft_llama30b/",
    "30b_sft_ppo": "output/ContextualAI/archangel_sft_ppo_llama30b/",
    "30b_sft_dpo": "output/ContextualAI/archangel_sft_dpo_llama30b/",
    "30b_sft_kto": "output/ContextualAI/archangel_sft_kto_llama30b/",
    "llama2-7b-chat": "output/meta-llama/Llama-2-7b-chat-hf/",
    "llama2-13b-chat": "output/meta-llama/Llama-2-13b-chat-hf/",
    "llama2-70b-chat": "output/llama2-70b-chat/",
    "olmo-7b-sft": "output/olmo-7b-sft/",
    "olmo-7b-instruct": "output/olmo-7b-instruct/",
    "mistral-chat": "output/mistralai/Mistral-7B-Instruct-v0.2/",
    "gpt-3.5-turbo-0125": "output/gpt-3.5-turbo-0125/",
    "gpt4": "output/gpt4/"
}


file_suffixes = ["etiquette_none_conditioned", "etiquette_country_conditioned", 
                 "etiquette_value_conditioned", "etiquette_rot_conditioned"]

subaxes = ["basic_etiquette", "eating", "visiting", "gift_giving"]
results_df = pd.DataFrame()
count = 0
for j, suffix in enumerate(file_suffixes): # degree
    results_df = pd.DataFrame()
    count = 0
    for i, prefix in enumerate(file_prefixes.keys()): # model
    
        file_path = os.path.join(file_prefixes[prefix], suffix + "_" + prefix + ".csv")
        if not os.path.exists(file_path):
            continue
        data = pd.read_csv(file_path)
        
        for k, axis in enumerate(subaxes): # subaxes
            subset = data.loc[data['Subaxis'] == axis]
            
            if 'prediction_label_temp0.0' in subset:
                col = 'prediction_label_temp0.0'
            else:
                col = 'prediction_label_temp0.001'
            accuracy = accuracy_score(subset['Gold Label'], subset[col])
            f1 = f1_score(subset['Gold Label'], subset[col], average='micro')
            precision = precision_score(subset['Gold Label'], subset[col], average='micro')
            recall = recall_score(subset['Gold Label'], subset[col], average='micro')

            results_df.loc[count, "model"] = prefix 
            results_df.loc[count, "axis"] = axis
            results_df.loc[count, "type"] =  suffix.replace("etiquette_", "")
            results_df.loc[count, "accuracy"] = accuracy
            results_df.loc[count, "precision"] = precision
            results_df.loc[count, "recall"] = recall
            results_df.loc[count, "f1"] = f1
            count += 1

    results_df.to_csv("output_subaxes/{}_results_micro.csv".format(suffix.replace("etiquette_", "")))
    print(results_df)
print(results_df)






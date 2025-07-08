import pandas as pd
import numpy as np
import matplotlib as mpl
import os

data = pd.read_csv("output_bins/value_conditioned_results_micro.csv")


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
results = pd.DataFrame()
for i, prefix in enumerate(file_prefixes.keys()):
    subset = data.groupby('model').get_group(prefix)
    max_acc_bin = subset.loc[subset['accuracy'].idxmax()]
    min_acc_bin = subset.loc[subset['accuracy'].idxmin()]
    acc_diff = max_acc_bin['accuracy'] - min_acc_bin['accuracy']

    results.loc[i, 'model'] = prefix
    results.loc[i, 'max_bin'] = max_acc_bin['bin']
    results.loc[i, 'max_bin_acc'] = max_acc_bin['accuracy']
    results.loc[i, 'min_bin'] = min_acc_bin['bin']
    results.loc[i, 'min_bin_acc'] = min_acc_bin['accuracy']
    results.loc[i, 'diff'] = acc_diff
    results.loc[i, 'overall_acc'] = subset['accuracy'].mean()

results.to_csv("output_bins/country_model_bin_min_max_micro.csv")

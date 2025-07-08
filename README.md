# NormAd
Code repo for ["NORMAD: A Framework for Measuring the Cultural Adaptability of Large Language Models"](https://arxiv.org/pdf/2404.12464). 

# Abstract 

To be effectively and safely deployed to global user populations, large language models (LLMs) may need to adapt outputs to user values and cultures, not just know about them. We introduce NormAd, an evaluation framework to assess LLMs' cultural adaptability, specifically measuring their ability to judge social acceptability across varying levels of cultural norm specificity, from abstract values to explicit social norms. As an instantiation of our framework, we create NormAd-Eti, a benchmark of 2.6k situational descriptions representing social-etiquette related cultural norms from 75 countries. Through comprehensive experiments on NormAd-Eti, we find that LLMs struggle to accurately judge social acceptability across these varying degrees of cultural contexts and show stronger adaptability to English-centric cultures over those from the Global South. Even in the simplest setting where the relevant social norms are provided, the best LLMs' performance (< 82\%) lags behind humans (> 95\%). In settings with abstract values and country information, model performance drops substantially (< 60\%), while human accuracy remains high (> 90\%). Furthermore, we find that models are better at recognizing socially acceptable versus unacceptable situations. Our findings showcase the current pitfalls in socio-cultural reasoning of LLMs which hinder their adaptability for global audiences. 

## HuggingFace Dataset 
```
from datasets import load_dataset

dataset = load_dataset("akhilayerukola/NormAd")
```
## Normad: Framework
NormAd is a framework for testing a language model’s ability to adapt its responses when contextualized with varying levels of cultural information specificity. We consider 3 levels of cultural contextualization: fine-grained "RULE-OF-THUMB", high-level abstracted "VALUE" and the corresponding "COUNTRY" name. 

## Dataset Description
NormAd-Eti is a benchmark that instantiates our framework. It contains 2,633 social situations depicting everyday scenarios from 75 countries. Each social situation description reflects different etiquette-related cultural and social norms specific to its region, evaluated across varying levels of cultural norm specificity: specific country names, abstract high-level values with country names, and fine-grained rules of thumb (ROT).

![Figure providing a snapshot of the dataset](assets/overview_figure.png)

## Dataset Construction
Our NormAd-Eti construction pipeline consists of 4 parts: 

&nbsp;a) Generation: We source social etiquette-related social norms from [Cultural Atlas](https://culturalatlas.sbs.com.au/) and systematically transform them into grounded social situation description, ROT, and VALUE 

&nbsp;b) Filtration: We perform three rounds of automatic filtering and sanity checks to eliminate inconsistencies 

&nbsp;c) Validation: We conduct extensive human validation of the constructed dataset 

&nbsp;d) Human Performance: We conduct a small-scale assessment of human performance.

![Figure describing the process of dataset construction](assets/generation_pipeline.png)

# Directory Structure
```
├── .gitattributes
├── .gitignore
├── .vscode
│   ├── settings.json
├── README.md
├── bad_prompt_collections.txt # error reporting
├── bad_scrape.txt # error reporting
├── conf             # configs for data collection
│   ├── __init__.py
│   ├── config.yaml
│   ├── prompts
│   │   ├── prompts.yaml
│   ├── story_inference
│   │   ├── story_inference.yaml
│   ├── webscrape
│   │   ├── webscrape.yaml
├── data_and_heval # final datasets and human eval sampling
│   ├── datasets **[EXTRACTED]** REFER TO # DATA SECTION for how to
│   │   ├── Etiquette_gpt4_3opt_filtered.csv
│   │   ├── normad_etiquette_final_data.csv # the final dataset for normad
│   │   ├── country_iwlist.csv
│   │   ├── full_data_gpt4_3options.csv
|   |   ├── 480_human_subset.csv
│   ├── human_eval_inhouse
│   │   ├── generate_iwsubset.py
│   ├── human_eval_mturk
│   │   ├── sample_across_label.py
├── environment.yml # conda environment
├── output
│   ├── .gitkeep
├── requirements.txt
├── src
│   ├── __init__.py
│   ├── analysis
│   │   ├── .gitignore
│   │   ├── __init__.py
│   │   ├── analyze_resp_etiquette.py
│   │   ├── analyze_stddev.py
│   │   ├── country_iwlist.csv
│   │   ├── csv_merge.py
│   │   ├── get_cultural_bin_differences.py
│   │   ├── get_model_scores_iw.py
│   │   ├── get_model_scores_overall.py
│   │   ├── get_model_scores_subaxes.py
│   │   ├── group_mapping.csv
│   │   ├── model_outputs.zip
│   │   ├── modelwise_plots.py
│   │   ├── plots_alignment.ipynb
│   │   ├── plots_bins.ipynb
│   │   ├── plots_overall.ipynb
│   │   ├── plots_updated.ipynb
│   │   ├── results_micro.csv
│   │   ├── utils.py
│   ├── base.py
│   ├── data_description.py
│   ├── model
│   │   ├── __init__.py
│   │   ├── base_model.py
│   │   ├── base_model_old.py
│   │   ├── llama_tgi_model.py
│   │   ├── llama_tgi_model_old.py
│   │   ├── llama_vllm_model.py
│   │   ├── mistral_model.py
│   │   ├── mistral_tgi_model.py
│   │   ├── olmo.py
│   │   ├── olmo_model.py
│   │   ├── openai_model.py
│   ├── response_collection
│   │   ├── __init__.py
│   │   ├── dqe.py
│   │   ├── run_inference.sh
│   │   ├── run_model_inference.py
│   │   ├── utils.py
│   │   ├── utils_2.py
│   │   ├── utils_cot.py
│   ├── sbatch_scripts
│   │   ├── aya.sh
│   │   ├── llama_13.sh
│   │   ├── llama_13_chat.sh
│   │   ├── llama_7.sh
│   │   ├── llama_70_chat.sh
│   │   ├── llama_7_chat.sh
│   │   ├── mistral_7.sh
│   │   ├── olmo_7b_instruct.sh
│   │   ├── olmo_7b_sft.sh
│   │   ├── sbatch_gpt3.5.sh
│   │   ├── sbatch_l13_response.sh
│   │   ├── sbatch_l7_response.sh
│   │   ├── sbatch_olmo_ins_inference.sh
│   │   ├── sbatch_olmo_sft_inference.sh
│   ├── story_collection
│   │   ├── __init__.py
│   │   ├── collect_irrelevant.py
│   │   ├── collect_stories.py
│   │   ├── estimate_cost.py
│   │   ├── few_shots_v2.py
│   │   ├── few_shots_v3.py
│   │   ├── get_csv.py
│   │   ├── gpt4_filter_neutral.py
│   │   ├── run_model_validation_stage1_rot.py
│   │   ├── run_model_validation_stage2_fix_rot.py
│   │   ├── sbatch_irrel.sh
│   │   ├── utils.py
│   ├── webscrape
│   │   ├── __init__.py
│   │   ├── webscrape.py
│   │   ├── webscrape_utils.py
├── step2_output
│   ├── .gitkeep
├── story_prompts
│   ├── v1
│   │   ├── guidelines.txt
│   │   ├── prefix.txt
│   │   ├── prompt-cwvs.txt
│   │   ├── prompt_ca.txt
│   │   ├── reminder.txt
│   ├── v2
│   │   ├── guidelines_affirm.txt
│   │   ├── guidelines_irrelevant.txt
│   │   ├── guidelines_negate.txt
│   │   ├── prefix.txt
│   │   ├── prefix_irrelevant.txt
│   │   ├── reminder.txt
```
# Installation 
## Create a virtualenv and install dependencies:
```
$ python -m venv path/to/venv 
$ source path/to/venv/bin/activate
$ pip install -r requirements.txt
```
OR conda 
```
$ conda create -f environment.yml
```

# Configuration 
Parameter passing is handled by the `hydra` library. All configuration files are in the `conf/` directory. 
- Make sure to override any calls with `hydra/job_logging=disabled` and `hydra/hydra_logging=disabled` if you want all logs to be shut off (this is also present in `conf/config.yaml`).

# Data

Data is protected by [easy-dataset-share](https://github.com/Responsible-Dataset-Sharing/easy-dataset-share). Make sure to run `easy-dataset-share magic-unprotect-dir /path/to/normad/data_and_heval/datasets.zip.enc -p normad-benchmark-data -rc` to get the data. 

# Generating your own Normad! (For axes other than Etiquette)
- `python -m path.to.file.without.extension`.
- Please check the steps below for filenames and paths.

## Step 0: Webscraping
- This project uses the `BeautifulSoup4` library to scrape the cultural atlas dataset. Please check out the `src/webscrape/webscrape.py` file for scraping. 
- Make sure to specify the axes needed for scraping in the config. Currently, only the `Etiquette` axis has been tested. 
- The scraped output is saved in the `data/` directory in json and markdown. The directory can be customized by changing the config values in the `conf/webscrape` folder.
- Some of the scraped data is stored in the `bad_scrape.txt` file. This is a list of countries that have been scraped incorrectly.
- Such countries will need to be manually handled 

## Step 1: Story generation
- We're currently using GPT-3.5 to generate stories based on the cultural atlas background. Please check `src/story_collection/collect_story.py`.
- Make sure to enter the path to the OpenAI token in the config (`conf/prompts/prompts.yaml`)
- The prompt has been divided into prefix, guidelines and reminder. It is very verbosely coded as of now, in separate files.
- The outputs are stored as a `csv` in the `outputs` directory. This can be modified in the `conf/prompts/prompts.yaml` config file.

## Step 2: Story response curation
- Please check `src/response_collection/run_model_inference.py`. 
- Sample command present in `src/response_collection/run_inference.sh`.

## Step 3: Analysis
- Presents plots for Models, subaxes, and Inglehart-Welzel bins. Please check `src/analysis/analyze_resp_etiquette.py`.
- The config file is the same as that of Step 2 (`conf/story_inference/story_inference.yaml`), and whatever configurations have been marked `True` (out of `no_condition`, `country_condition`, `cval_condition`, `rot_condition`) will be automatically plotted.

# License

Shield: [![CC BY 4.0][cc-by-shield]][cc-by]

This work is licensed under a
[Creative Commons Attribution 4.0 International License][cc-by].

[![CC BY 4.0][cc-by-image]][cc-by]

[cc-by]: http://creativecommons.org/licenses/by/4.0/
[cc-by-image]: https://i.creativecommons.org/l/by/4.0/88x31.png
[cc-by-shield]: https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg

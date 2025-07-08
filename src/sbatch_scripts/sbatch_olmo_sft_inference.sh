#!/bin/sh 
#SBATCH --partition=general
#SBATCH --mem=16Gb
#SBATCH -t 08:00:00              # time limit: (D-HH:MM) 
#SBATCH --job-name=o7sft-inference
#SBATCH --error=/home/abhinavr/logs/o7sft-infer.%j.err
#SBATCH --output=/home/abhinavr/logs/o7sft-infer.%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=abhinavr@andrew.cmu.edu

mkdir -p /scratch/abhinavr
source /home/abhinavr/git_clones/culEval/bin/activate

REPO_DIR=/home/abhinavr/git_clones/RLKF/
export HUGGING_FACE_HUB_TOKEN=$(cat $REPO_DIR/key/hf_key.txt)
export HF_HOME=/data/tir/projects/tir5/users/abhinavr/hf_home
export HF_DATASETS_CACHE=/data/tir/projects/tir5/users/abhinavr/hf_datasets_cache

cd $REPO_DIR
# VERIFY CONFIG VALUES FIRST
python -m src.response_collection.get_response story_inference.block_user_input=false story_inference.model_name="allenai/OLMo-7B-SFT" story_inference.model_url="http://inst-0-35:8088" story_inference.no_condition=true story_inference.country_condition=true
# alternate cache path
#/data/tir/projects/tir5/users/abhinavr/hf_home/hub
#/data/tir/projects/tir2/models/tgi_cache/hub/

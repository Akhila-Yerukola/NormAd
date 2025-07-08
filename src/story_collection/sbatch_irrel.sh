#!/bin/sh 
#SBATCH --partition=general
#SBATCH --mem=16Gb
#SBATCH -t 08:00:00              # time limit: (D-HH:MM) 
#SBATCH --job-name=post_filter
#SBATCH --error=/home/abhinavr/logs/post_filter.%j.err
#SBATCH --output=/home/abhinavr/logs/post_filter.%j.out
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
python -m src.story_collection.gpt4_filter_neutral
# alternate cache path
#/data/tir/projects/tir5/users/abhinavr/hf_home/hub
#/data/tir/projects/tir2/models/tgi_cache/hub/

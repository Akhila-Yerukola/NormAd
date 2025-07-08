#!/bin/sh 
#SBATCH --gres=gpu:A6000:2
#SBATCH --partition=general
#SBATCH --mem=32Gb
#SBATCH -t 2-00:00:00              # time limit: (D-HH:MM) 
#SBATCH --job-name=llama-7-nonchat
#SBATCH --error=/home/abhinavr/logs/llama-7.%j.err
#SBATCH --output=/home/abhinavr/logs/llama-7.%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=abhinavr@andrew.cmu.edu

mkdir -p /scratch/abhinavr
source /home/abhinavr/miniconda3/etc/profile.d/conda.sh
conda activate fern-tgi
#conda activate /data/tir/projects/tir6/general/pfernand/conda/envs/tgi-env-public
REPO_DIR=/home/abhinavr/git_clones/RLKF/
export HUGGING_FACE_HUB_TOKEN=$(cat $REPO_DIR/key/hf_key.txt)
export HF_HOME=/data/tir/projects/tir7/user_data/abhinavr/hf_home
export HF_DATASETS_CACHE=/data/tir/projects/tir7/user_data/abhinavr/hf_datasets_cache

cd /home/abhinavr/git_clones/text-generation-inference
text-generation-launcher --model-id meta-llama/Llama-2-7b-hf  --port 8088 --master-port 41024 --huggingface-hub-cache /data/tir/projects/tir7/user_data/abhinavr/hf_home/hub/ --shard-uds-path /scratch/abhinavr/tgi-uds-socket-2 
# alternate cache path
#/data/tir/projects/tir5/users/abhinavr/hf_home/hub
#/data/tir/projects/tir2/models/tgi_cache/hub/

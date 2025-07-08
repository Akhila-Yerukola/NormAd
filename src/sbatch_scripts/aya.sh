#!/bin/sh 
#SBATCH --gres=gpu:A6000:2
#SBATCH --partition=long
#SBATCH --mem=32Gb
#SBATCH -t 5-00:00:00              # time limit: (D-HH:MM) 
#SBATCH --job-name=aya-101
#SBATCH --error=/home/abhinavr/logs/aya-101.%j.err
#SBATCH --output=/home/abhinavr/logs/aya-101.%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=abhinavr@andrew.cmu.edu

mkdir -p /scratch/abhinavr
source ~/miniconda3/etc/profile.d/conda.sh
conda activate fern-tgi
#conda activate /data/tir/projects/tir6/general/pfernand/conda/envs/tgi-env-public
# open key.txt and copy the key to the next line
REPO_DIR=/home/abhinavr/git_clones/RLKF/
export HUGGING_FACE_HUB_TOKEN=$(cat $REPO_DIR/key/hf_key.txt)
export HF_HOME=/data/tir/projects/tir7/user_data/abhinavr/hf_home
export HF_DATASETS_CACHE=/data/tir/projects/tir7/user_data/abhinavr/hf_datasets_cache
cd /home/abhinavr/git_clones/text-generation-inference
text-generation-launcher --model-id CohereForAI/aya-101 --port 9104 --master-port 23457 --huggingface-hub-cache /data/tir/projects/tir5/users/abhinavr/hf_home/hub/ --quantize bitsandbytes --shard-uds-path /scratch/abhinavr/tgi-uds-socket-2 
# alternate cache path
#/data/tir/projects/tir5/users/abhinavr/hf_home/
#/data/tir/projects/tir2/models/tgi_cache/hub/
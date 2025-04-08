#!/bin/bash
#SBATCH -p 002-partition-vlm
#SBATCH --job-name=DeCapBench
#SBATCH -o outputs/slurm-%j.out
#SBATCH -e outputs/slurm-%j.out
#SBATCH --gres=gpu:1

MODEL=${1:-/lustre/share/downloaded/models/Qwen/Qwen2.5-VL-7B-Instruct/}
PROMPT=${2:-"Generate a detailed and coherent description of the entire scene by explaining every visible detail in the image, including each element's appearance, function, and interrelationship."}
source ../vlm_VG_caption/.venv_VG/bin/activate

python generate_QwenModel.py \
    --model_path "$MODEL" \
    --image_dir "./images" \
    --prompt "${PROMPT}"

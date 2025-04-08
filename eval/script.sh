#!/bin/bash
#SBATCH -p 002-partition-vlm
#SBATCH --job-name=DeCapBench
#SBATCH -o outputs/slurm-%j.out
#SBATCH -e outputs/slurm-%j.out
#SBATCH --gres=gpu:2

# Usage: bash script.sh INPUT_PATH RESULT_DIR


INPUT_PATH=${1:-"./generated_Qwen2.5-VL-3B-Instruct.jsonl"}
RESULT_DIR=${2:-"./eval_results/"}

IMAGE_DIR="./images"
ANNOTATION_PATH="./eval/human_annotated_oracle_facts.jsonl"

EVAL_DIR=$RESULT_DIR/eval_details
EVAL_FINAL=$RESULT_DIR/eval_metric.txt

API="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"

source .venv_DeCap/bin/activate

mkdir -p $EVAL_DIR

python3 eval/evaluation_decapbench.py --image_path $IMAGE_DIR --caption_file $INPUT_PATH --annotation_file $ANNOTATION_PATH --save_folder $EVAL_DIR --api_key "$API" --part_idx 0 --n_parts 2 &
python3 eval/evaluation_decapbench.py --image_path $IMAGE_DIR --caption_file $INPUT_PATH --annotation_file $ANNOTATION_PATH --save_folder $EVAL_DIR --api_key "$API" --part_idx 1 --n_parts 2 &


python3 eval/compute_results.py --eval_folder $EVAL_DIR | tee "$EVAL_FINAL"

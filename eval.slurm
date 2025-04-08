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


INPUT_PATH="./generated_${MODEL}.jsonl"
RESULT_DIR="./eval_results/"

IMAGE_DIR="./images"
ANNOTATION_PATH="./eval/human_annotated_oracle_facts.jsonl"
mkdir -p $RESULT_DIR/$MODEL
EVAL_DIR=$RESULT_DIR/$MODEL/eval_details
EVAL_FINAL=$RESULT_DIR/$MODEL/eval_metric.txt


source .venv_DeCap/bin/activate

mkdir -p $EVAL_DIR

python3 eval/evaluation_decapbench.py --image_path $IMAGE_DIR --caption_file $INPUT_PATH --annotation_file $ANNOTATION_PATH --save_folder $EVAL_DIR --api_key "$API" --part_idx 0 --n_parts 1

python3 eval/compute_results.py --eval_folder $EVAL_DIR | tee "$EVAL_FINAL"

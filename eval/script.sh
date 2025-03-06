#!/bin/bash
# Usage: bash script.sh INPUT_PATH RESULT_DIR


INPUT_PATH=$1
RESULT_DIR=${2:-"./eval_results/"}

IMAGE_DIR="./images"
ANNOTATION_PATH="./human_annotated_oracle_facts.jsonl"

EVAL_DIR=$RESULT_DIR/eval_details
EVAL_FINAL=$RESULT_DIR/eval_metric.txt

mkdir -p $EVAL_DIR

python3 evaluation_decapbench.py --image_path $IMAGE_DIR --caption_file $INPUT_PATH --annotation_file $ANNOTATION_PATH --save_folder $EVAL_DIR --api_key "xxxx" --part_idx 0 --n_parts 5 &
python3 evaluation_decapbench.py --image_path $IMAGE_DIR --caption_file $INPUT_PATH --annotation_file $ANNOTATION_PATH --save_folder $EVAL_DIR --api_key "xxxxx" --part_idx 1 --n_parts 5 &
python3 evaluation_decapbench.py --image_path $IMAGE_DIR --caption_file $INPUT_PATH --annotation_file $ANNOTATION_PATH --save_folder $EVAL_DIR --api_key "xxxxx" --part_idx 2 --n_parts 5 &
python3 evaluation_decapbench.py --image_path $IMAGE_DIR --caption_file $INPUT_PATH --annotation_file $ANNOTATION_PATH --save_folder $EVAL_DIR --api_key "xxxx" --part_idx 3 --n_parts 5 &
python3 evaluation_decapbench.py --image_path $IMAGE_DIR --caption_file $INPUT_PATH --annotation_file $ANNOTATION_PATH --save_folder $EVAL_DIR --api_key "xx" --part_idx 4 --n_parts 5 &

python3 compute_results.py --eval_folder $EVAL_DIR | tee "$EVAL_FINAL"
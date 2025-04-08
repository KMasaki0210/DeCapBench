import os
import io
import json
import argparse
import base64
from PIL import Image
from tqdm import tqdm

import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

def pil2base64(pil_img):
    buf = io.BytesIO()
    pil_img.save(buf, format='jpeg')
    image_bytes = buf.getvalue()
    return base64.b64encode(image_bytes).decode('utf-8')

def main():
    parser = argparse.ArgumentParser(description="Generate image captions and save as JSONL")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model directory")
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt for caption generation")
    parser.add_argument("--image_dir", type=str, required=True, help="Directory containing images")
    parser.add_argument("--save_file", type=str, default=None, help="Output JSONL file name (optional)")
    args = parser.parse_args()

    # Initialize model and processor
    print("Loading model from", args.model_path)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path, torch_dtype="auto", device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(args.model_path, use_fast=False)

    model_name = os.path.basename(os.path.normpath(args.model_path))
    output_filename = args.save_file if args.save_file else f"generated_{model_name}.jsonl"

    image_files = [f for f in os.listdir(args.image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    print("Found", len(image_files), "images.")

    results = []
    device = "cuda" if torch.cuda.is_available() else "cpu"

    for img_file in tqdm(image_files, desc="Generating captions"):
        img_path = os.path.join(args.image_dir, img_file)
        try:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": f"file://{os.path.abspath(img_path)}",
                        },
                        {"type": "text", "text": args.prompt},
                    ],
                }
            ]
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to(device)
            generated_ids = model.generate(**inputs, max_new_tokens=512)
            generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
            output_text = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            caption = output_text[0].strip()
        except Exception as e:
            print(f"Error processing {img_file}: {e}")
            caption = ""

        results.append({
            "image_name": img_file,
            "caption": caption
        })

    # 保存
    with open(output_filename, "w", encoding="utf-8") as f:
        for record in results:
            json.dump(record, f, ensure_ascii=False)
            f.write("\n")
    print("Saved generated captions to", output_filename)

if __name__ == "__main__":
    main()

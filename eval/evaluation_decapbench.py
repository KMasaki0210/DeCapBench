import io
import os
import re
import json
import base64
import argparse
from collections import defaultdict
import copy

import ast
from tqdm import tqdm
import numpy as np
from PIL import Image
import openai
import torch
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
    retry_if_not_exception_type
)
import traceback


def load_jsonl(filename):
    with open(filename, "r", encoding="utf-8") as f:
        return [json.loads(l.strip("\n")) for l in f.readlines()]

def save_jsonl(data, filename):
    with open(filename, "w") as outfile:
        for idx, element in enumerate(data):
            json.dump(element, outfile, ensure_ascii=False)
            outfile.write("\n")


def pil2base64(pil_img):
    buf = io.BytesIO()
    # Save the image as a PNG to the buffer
    pil_img.save(buf, format='jpeg')
    # Retrieve the byte data
    image_bytes = buf.getvalue()
    # Encode as base64
    image_base64 = base64.b64encode(image_bytes)
    # Convert bytes to string
    image_base64_str = image_base64.decode('utf-8')
    return image_base64_str


def parse_text_json(text):
    try:
        text_ = text.split('```json')[-1].split('```')[0]
        text_ = text_.strip('\n')
        return ast.literal_eval(text_)
    except:
        pass

    try:
        text_ = text.split('```json')[-1].strip('`').strip()
        text_ = text_.strip('\n').replace('\n', '')
        return ast.literal_eval(text_)
    except:
        pass

    try:
        text_ = text.split('```json')[-1].strip('`').strip()
        text_ = text_.strip('\n').replace('\n', '')
        return json.loads(text_)
    except:
        pass

    try:
        text_ = text.split('```json')[-1].strip('`').strip()
        text_ = text_.strip('\n').replace('\n', '').replace('`', '')
        return json.loads(text_)
    except:
        return None


def text_message_creator(prompt, sys_prompt="You are a helpful assistant."):
    system_msg = {"role": "system", "content": sys_prompt}
    user_msg = {"role": "user", "content": prompt}
            
    return [system_msg, user_msg]

def run_gpt_text(client, prompt, temperature=1.0, sys_prompt="You are a helpful assistant.", n=1):
    msgs = text_message_creator(prompt, sys_prompt=sys_prompt)

    @retry(retry=retry_if_not_exception_type(openai.BadRequestError), wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
    def completion_with_backoff(**kwargs):
        return client.chat.completions.create(**kwargs)

    completion = completion_with_backoff(
        model="gpt-4o-2024-05-13",
        messages=msgs,
        max_tokens=2000,
        temperature=temperature,
        n=n,
    )
    results = [choice.message.content for choice in completion.choices]
    
    return results, completion

def multimodal_message_creator(prompt, image_path, sys_prompt="You are a helpful assistant.", detail='high'):
    system_msg = {"role": "system", "content": sys_prompt}
    user_content = []
    image = Image.open(image_path).convert('RGB') 
    base64_img = pil2base64(image)
    new_msg = {
        "type": "image_url", 
        "image_url": {
            "url": f"data:image/jpeg;base64,{base64_img}", 
            "detail": detail
        }
    }
    user_content.append(new_msg)
    user_content.append({"type": "text", "text": prompt})
    user_msg = {"role": "user", "content": user_content}
    return [system_msg, user_msg]


def run_gpt_multimodal(client, prompt, image_path, temperature=1.0, sys_prompt="You are a helpful assistant.", n=1, detail='high'):
    msgs = multimodal_message_creator(prompt, image_path, sys_prompt=sys_prompt, detail=detail)
    @retry(retry=retry_if_not_exception_type(openai.BadRequestError), wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
    def completion_with_backoff(**kwargs):
        return client.chat.completions.create(**kwargs)

    completion = completion_with_backoff(
        model="gpt-4o-2024-05-13",
        messages=msgs,
        max_tokens=2000,
        temperature=temperature,
        n=n,
    )
    results = [choice.message.content for choice in completion.choices]
    
    return results, completion


def format_json_input(input_list):
    res = '```json[\n'
    for d in input_list:
        res += '  {},\n'.format(json.dumps(d, ensure_ascii=False))
    res += ']\n```'
    return res

extract_prompt = \
'''
You are a lingustic expert in extracting atomic facts in the image caption. In specific, "Atomic facts" refer to the smallest standalone pieces of information that collectively represent the entire meaning of the sentence without losing any detail, which typically describe various properties of the visual elements in an image. The atomic fact should be a simple statement. The fact must represent the smallest piece of information that cannot be further broken down without loss of meaning. Abstract concepts or broad interpretations should be reduced to more basic, constituent observations if possible. The atomic fact should only contain ONE primary element.
 
When extracting atomic facts from image caption, it is useful to assign unique identifiers to the primary objects or entities being discussed. This will help in maintaining clarity and preventing confusion, especially when there are multiple similar objects or entities. For example, if the caption mentions two cats, you can assign unique identifiers such as "cat_1" and "cat_2" to distinguish them. Besides, for each attribute, you should also assign the identifier to the object it belongs to. Meanwhile, for spatial relationships, you can assign the identifier to the object that is the subject of the relationship in the atomic fact.

For each atomic fact, you should also need to justify whether the atomic fact directly describe the image or not.

**IMPORTANT**: Please extract ALL of the atomic facts in the image caption. DO NOT omit any information! 

The output should be a list of dict [{{"fact": [ATOMIC FACT], "identifier": [UNIQUE ID], "relevance": 1/0}}, ...] into JSON format. The "identifier" would be optional, if the item in the fact has already been identified with ids. The "relevance" would be 1 if the atomic fact directly describe the content of the image. Otherwise it would be 0 if the atomic fact is inference or extension to the description and not directly describe to the content of image.

Here are some examples:
### Caption
The image features a close-up view of a delicious sub sandwich, with the bread halved to showcase its contents. The sandwich is loaded with various ingredients, including meat, lettuce, tomatoes, and pickles. There are several slices of tomato spread across the sandwich. The close-up perspective highlights the detailed texture and appearance of the ingredients.
### Atomic Facts
```json
[
   {{"fact": "There is a sub sandwich_0", "identifier": "sandwich_0", "relevance": 1}},
   {{"fact": "The sandwich_0 is delicious.", "relevance": 1}},
   {{"fact": "The image features a close-up view of sandwich_0.", "relevance": 1}},
   {{"fact": "The bread of sandwich_0 is halved.", "relevance": 1}},
   {{"fact": "The sandwich_0 is halved to showcase its contents.", "relevance": 1}},
   {{"fact": "The sandwich_0 is loaded with various ingredients.", "relevance": 1}},
   {{"fact": "The sandwich_0 is loaded with meat.", "relevance": 1}},
   {{"fact": "The sandwich_0 is loaded with lettuce.", "relevance": 1}},
   {{"fact": "The sandwich_0 is loaded with tomatoes", "relevance": 1}},
   {{"fact": "The sandwich_0 is loaded with pickles", "relevance": 1}},
   {{"fact": "There are several slices of tomato.", "relevance": 1}},
   {{"fact": "The slices of tomato spread across the sandwich_0.", "relevance": 1}},
   {{"fact": "The close-up perspective highlights the detailed texture of the ingredients in the sandwich_0.", "relevance": 0}},
   {{"fact": "The close-up perspective highlights the appearance of the ingredients in the sandwich_0.", "relevance": 0}}
]
```


### Caption
An overhead shot of a light gray concrete surface with the words "PLEASE WALK BIKES" written in dark green stencil letters. The words are vertically stacked on top of each other, with "PLEASE" at the top and "WALK" at the bottom. The letters are outlined in a dark green color.
### Atomic Facts
```json
[
   {{"fact": "The image shows an overhead shot of surface_0.", "identifier": "surface_0", "relevance": 1}},
   {{"fact": "The surface_0 in the image is light gray.", "relevance": 1}},
   {{"fact": "The surface_0 in the image is concrete.", "relevance": 1}},
   {{"fact": "The words_0 'PLEASE WALK BIKES' are written on the surface_0.", "identifier": "words_0", "relevance": 1}},
   {{"fact": "The words_0 are written in dark green letters.", "relevance": 1}},
   {{"fact": "The words_0 are written in stencil letters", "relevance": 1}},
   {{"fact": "The words_0 are vertically stacked on top of each other.", "relevance": 1}},
   {{"fact": "The word 'WALK' is in the middle.", "relevance": 1}},
   {{"fact": "The word 'PLEASE' is at the top.", "relevance": 1}},
   {{"fact": "The word 'BIKES' is at the bottom.", "relevance": 1}},
   {{"fact": "The letters of words_0 are outlined in dark green color.", "relevance": 1}}
]
```


### Caption
Three statues of the Statue of Liberty are displayed in a museum, captured from an eye-level perspective. The central statue commands attention, standing proudly on a black pedestal. Its right arm is elevated, a torch held high, while the left hand clutches a book. The statue's attire is a flowing robe, cinched at the waist with a sash, and adorned with a headdress featuring feathers. The statue's facial features are delicately carved, with a serene expression on its face. To the left of the central statue, a smaller model replicates the same pose. The statue's left arm is raised, a torch held high, while the right hand holds a book. 
### Atomic Fact
```json
[
   {{"fact": "There are three statues", "relevance": 1}},
   {{"fact": "The statues are Statue of Liberty.", "relevance": 1}},
   {{"fact": "The statues are displayed in a museum.", "relevance": 1}},
   {{"fact": "The image is captured from an eye-level perspective.", "relevance": 1}},
   {{"fact": "The central statue_0 commands attention.", "identifier": "statue_0", "relevance": 0}},
   {{"fact": "The statue_0 stands proundly.", "relevance": 1}},
   {{"fact": "The statue_0 stands on a pedestal_0.", "identifier": "pedestal_0", "relevance": 1}},
   {{"fact": "The pedestal_0 is black.", "relevance": 1}},
   {{"fact": "The statue_0 has its right arm elevated.", "relevance": 1}},
   {{"fact": "There is a torch_0", "identifier": "torch_0", "relevance": 1}},
   {{"fact": "The statue_0 holds a torch_0 high with its right arm.", "relevance": 1}},
   {{"fact": "The statue_0 clutches a book_0.", "identifier": "book_0", "relevance": 1}},
   {{"fact": "The statue_0 clutches a book_0 with its left hand.", "relevance": 1}},
   {{"fact": "The statue_0's attire is a robe_0", "identifier": "robe_0", "relevance": 1}},
   {{"fact": "The robe_0 is flowing.", "relevance": 1}},
   {{"fact": "The robe_0 is cinched at the waist.", "relevance": 1}},
   {{"fact": "The robe_0 has a sash.", "relevance": 1}},
   {{"fact": "The attire of statue_0 is adorned with a headdress_0.", "identifier": "headdress_0", "relevance": 1}},
   {{"fact": "The headdress_0 is a feathers.", "relevance": 1}},
   {{"fact": "The statue_0's facial features are delicately carved.", "relevance": 1}},
   {{"fact": "The statue_0 has a serene expression on its face.", "relevance": 1}},
   {{"fact": "There is a statue_1 to the left of statue_0.", "identifier": "statue_1", "relevance": 1}},
   {{"fact": "The statue_1 is smaller than statue_0.", "relevance": 1}},
   {{"fact": "The statue_1 replicates the same pose as statue_0.", "relevance": 1}},
   {{"fact": "The statue_1's left arm is raised.", "relevance": 1}},
   {{"fact": "There is a torch_1", "identifier": "torch_1", "relevance": 1}},
   {{"fact": "The statue_1 holds a torch_1 high with its left arm.", "relevance": 1}},
   {{"fact": "There is a book_1", "identifier": "book_1", "relevance": 1}},
   {{"fact": "The statue_1 holds a book_1.", "relevance": 1}},
   {{"fact": "The statue_1 holds a book_1 with its right hand.", "relevance": 1}}
]
```


### Caption
In the heart of the image, a white cat with black markings on its face and ears is the star of the scene. The cat is comfortably seated on a wooden table, its gaze directed towards the camera, exuding an air of curiosity and alertness. To the cat's right, a beige lamp with a white shade stands tall. The lamp, with its warm and inviting light, adds a cozy touch to the scene. On the left side of the table, a white notebook lies open, its pages filled with unseen stories. The notebook, with its promise of untold tales, adds an element of mystery to the image. The backdrop to this tableau is a white wall adorned with wooden paneling, providing a neutral canvas that allows the objects on the table to stand out. The overall composition of the image suggests a quiet moment of solitude, with the cat, lamp, and notebook each playing their part in this tranquil tableau.
### Atomic Facts
```json
[
   {{"fact": "There is a cat_0 in the image.", "identifier": "cat_0", "relevance": 1}},
   {{"fact": "The cat_0 is white.", "relevance": 1}},
   {{"fact": "The cat_0 has markings.", "relevance": 1}},
   {{"fact": "The markings of cat_0 are black.", "relevance": 1}},
   {{"fact": "The markings of cat_0 are on its face.", "relevance": 1}},
   {{"fact": "The markings of cat_0 are on its ears.", "relevance": 1}},
   {{"fact": "The cat_0 is the star of the scene.", "relevance": 0}},
   {{"fact": "The cat_0 is comfortably seated.", "relevance": 1}},
   {{"fact": "The cat_0 is seated on a table_0.", "identifier": "table_0", "relevance": 1}},
   {{"fact": "The table_0 is wooden.", "relevance": 1}},
   {{"fact": "The cat_0 is gazing towards the camera.", "relevance": 1}},
   {{"fact": "The cat_0 exudes an air of curiosity.", "relevance": 0}},
   {{"fact": "The cat_0 exudes an air of alertness.", "relevance": 0}},
   {{"fact": "There is a lamp_0.", "identifier": "lamp_0", "relevance": 1}},
   {{"fact": "The lamp_0 is to the cat_0's right.", "relevance": 1}},
   {{"fact": "The lamp_0 is beige.", "relevance": 1}},
   {{"fact": "The lamp_0 has a white shade.", "relevance": 1}},
   {{"fact": "The shade of lamp_0 stands tall.", "relevance": 1}},
   {{"fact": "The light from lamp_0 is warm and inviting.", "relevance": 1}},
   {{"fact": "The lamp_0 adds a cozy touch to the scene.", "relevance": 0}},
   {{"fact": "There is a notebook_0.", "identifier": "notebook_0"}},
   {{"fact": "The notebook_0 is on the left side of the table_0.", "relevance": 1}},
   {{"fact": "The notebook_0 is white.", "relevance": 1}},
   {{"fact": "The notebook_0 lies open.", "relevance": 1}},
   {{"fact": "The pages of notebook_0 is filled with unseen stories.", "relevance": 1}},
   {{"fact": "The notebook_0 has promise of untold tales.", "relevance": 0}},
   {{"fact": "The notebook_0 adds an element of mystery to the image.", "relevance": 0}},
   {{"fact": "The backdrop of the image is a wall_0.", "identifier": "wall_0", "relevance": 1}},
   {{"fact": "The wall_0 is white.", "relevance": 1}},
   {{"fact": "The wall_0 is adorned with paneling.", "relevance": 1}},
   {{"fact": "The paneling of wall_0 is wooden.", "relevance": 1}},
   {{"fact": "The wall_0 provides a neutral canvas for the objects on the table_0 to stand out.", "relevance": 0}},
   {{"fact": "The composition of the image suggests a quiet moment of solitude.", "relevance": 0}},
   {{"fact": "The cat_0, lamp_0, and notebook_0 each play a part in this tranquil tableau.", "relevance": 0}}
]
```


### Caption
This image features a close-up of a cylindrical container with a label that reads ""Homer's All- Purpose 5 Gal bucket."" The label has a white background with bold, orange text and a red border. The word ""Homer's"" is at the top of the label, with ""All- Purpose"" directly below it in a larger font. The size of the bucket, ""5 Gal,"" is mentioned at the bottom of the label.

On the left side of the label, there is a graphic of a red bucket with a white handle and a white spout. Above the bucket, there is a small orange and white logo that says ""The Home Depot,"" indicating the brand of the bucket. The background of the label features a light blue sky with a few white clouds.

The container itself is partially visible, with a red body and a silver top, suggesting that it is a paint bucket from The Home Depot. The focus of the image is on the label, with the container slightly blurred in the background. The label is positioned vertically, and the image is taken from a side angle, showing the label's front and partially its right side.
### Atomic Facts
```json
[
   {{"fact": "The image features a close-up of a container_0.", "identifier": "container_0", "relevance": 1}},
   {{"fact": "The container_0 is cylindrical.", "relevance": 1}},
   {{"fact": "There is a label_0 on the container_0.", "identifier": "label_0", "relevance": 1}},
   {{"fact": "The label_0 reads 'Homer's All-Purpose 5 Gal bucket.'", "relevance": 1}},
   {{"fact": "The label_0 has a white background.", "relevance": 1}},
   {{"fact": "The label_0 has bold text.", "relevance": 1}},
   {{"fact": "The label_0 has orange text.", "relevance": 1}},
   {{"fact": "The label_0 has a red border.", "relevance": 1}},
   {{"fact": "The word 'Homer's' is at the top of the label_0.", "relevance": 1}},
   {{"fact": "The word 'All-Purpose' is directly below 'Homer's' in a larger font.", "relevance": 1}},
   {{"fact": "The size '5 Gal' is mentioned at the bottom of the label_0.", "relevance": 1}},
   {{"fact": "There is a graphic of a bucket_0.", "identifier": "bucket_0", "relevance": 1}},
   {{"fact": "The bucket_0 is red.", "relevance": 1}},
   {{"fact": "The bucket_0 is on the left side of the label_0.", "relevance": 1}},
   {{"fact": "The bucket_0 has a white handle.", "relevance": 1}},
   {{"fact": "The bucket_0 has a white spout.", "relevance": 1}},
   {{"fact": "There is a small logo_0.", "identifier": "logo_0", "relevance": 1}},
   {{"fact": "The logo_0 is above the bucket_0.", "relevance": 1}},
   {{"fact": "The logo_0 is orange and white.", "relevance": 1}},
   {{"fact": "The logo_0 says 'The Home Depot.'", "relevance": 1}},
   {{"fact": "The logo_0 indicates the brand of the bucket_0.", "relevance": 0}},
   {{"fact": "The label_0's background features a light blue sky.", "relevance": 1}},
   {{"fact": "The label_0's background features a few white clouds.", "relevance": 1}},
   {{"fact": "The container_0 is partially visible.", "relevance": 1}},
   {{"fact": "The container_0 has a red body.", "relevance": 1}},
   {{"fact": "The container_0 has a silver top.", "relevance": 1}},
   {{"fact": "The container_0 suggests it is a paint bucket from The Home Depot.", "relevance": 0}},
   {{"fact": "The focus of the image is on the label_0.", "relevance": 1}},
   {{"fact": "The container_0 is slightly blurred in the background.", "relevance": 1}},
   {{"fact": "The label_0 is positioned vertically.", "relevance": 1}},
   {{"fact": "The image is taken from a side angle.", "relevance": 1}},
   {{"fact": "The image shows the front of the label_0.", "relevance": 1}},
   {{"fact": "The image shows partially the right side of the label_0.", "relevance": 1}}
]
```


### Caption
{caption}
### Atomic Facts
'''


verify_prompt = '''
You are an extraordinary visual-lingustic expert in verifying the correctness of a set of atomic facts given the image and the corresponding reference caption. The set of atomic facts are extracted from a paragraph of machine-generated image caption of that image. 

The set of atomic facts is represented as a list of dict [{{"fact": [ATOMIC FACT], "identifier": [UNIQUE ID]}}, ...] within JSON format. The identifier is unique and to identify the primary objects or entities being discussed. This will help in maintaining clarity and preventing confusion, especially when there are multiple similar objects or entities. For example, if the caption mentions two cats, we would assign unique identifiers such as "cat_1" and "cat_2" to distinguish them. Besides, for each attribute, it also assigned the identifier to the object it belongs to. Meanwhile, for spatial relationships, it assigned the identifier to the object that is the subject of the relationship in the atomic fact.

You should first go through all of the atomic facts, and understand the details and context presented within each atomic fact. Then you need to verify the correctness of each individual atomic facts by asking yourself: 
Statement: "[ATOMIC FACT]"\nDoes the statement correct according to image or reference caption? 

The output for the predicted set of atomic facts should be formatted as a list of dict as [{{"fact": [ATOMIC FACT], "identifier": [UNIQUE ID], "verification": 1/0}}, ...] in JSON format, where 1 represents the fact is correct and 0 represents the fact is incorrect. Other keys in the dictionary are the same as the input. The "identifier" would be optional, if the item in the fact has already been identified with ids as illustrated in the input.

>>> Reference Caption:
{reference_caption}

>>> Atomic Facts:
{atomic_facts}
'''


match_prompt = '''
You are now a visual-lingustic expert in matching two set of atomic facts generated from two captions.

You will be recieved a set of predicted atomic facts across a variety of categories and a set of oracle atomic facts (ground truth). The set of atomic facts is represented as a list of dict [{{"fact": [ATOMIC FACT], "identifier": [UNIQUE ID]}}, ...] within JSON format. In addition, each atomic fact in the oracle set would be companied with a unique "id" to identify the oracle atomic fact.

To match atomic facts from a predicted set in terms of the given image with oracle set of atomic facts. Here is the step by step instruction:
1. Preliminary Review: Conduct an initial review of both sets of atomic facts, considering all atomic facts. Understand the details and context presented within each atomic fact.
2. Inferring Identifier Mappings: Closely examine both sets to deduce potential correlations and mappings based on the content of the atomic facts. Determine if there are any unique identifiers or descriptors that hint at matching entities between the sets. For example, "cat_0" in the predicted set's atomic facts may be mapped to "cat_1" in the oracle set's atomic facts. Consider the attribute and spatial relation in both sets for possible mapping. Please note that there might be some attribute and spatial errors when mapping the objects. Try find the most similar mapping if exists (not need exact matching). If no oracle atomic fact matches, simply set matched oracle id to "None".

**IMPORTANT**: Please consider each atomic fact in the set individually, and MUST NOT omit any atomic facts from the predicted set.

You should only output the matching results which will be formatted as a list of dict as [{{"fact": [ATOMIC FACT], "identifier": [UNIQUE ID], "matched_oracle_id": [CORRESPONDING ORACLE ID]}}, ...] in JSON format. The "identifier" would be optional, if the item in the fact has already been identified with ids as illustrated in the predicted atomic facts. For key named "matched_oracle_id", the value of "matched_oracle_id" should be the corresponding "id" of the atomic fact in the oracle set. For the atomic fact in the predicted set which cannot be matched with any oracle atomic fact, set the value of "matched_oracle_id" to "None".

>>> Set of Atomic Facts:
{predicted_atomic_facts}

>>> Oracle Set of Atomic Facts:
{oracle_atomic_facts}

>>> Matching Result:
'''

def compute_precision_recall(predicted_atomic_facts, oracle_atomic_facts, verification_results, mappings):
    mapped_prediction = verification_results.copy()
    print('Mapping Num == Prediction Num:', set(mappings.keys()) == set([x['fact'] for x in mapped_prediction]))
    assert set(mappings.keys()) == set([x['fact'] for x in mapped_prediction])
    predict_ids = defaultdict(int)
    for d in mapped_prediction:
        d['matched_oracle_id'] = mappings[d['fact']]
        if d['matched_oracle_id'] is not None and d['matched_oracle_id'] != 'None':
            predict_ids[d['matched_oracle_id']] += 1

    predict_weights = {k: 1/v for k, v in predict_ids.items()}
    correct = 0
    total_atomic_facts = len(oracle_atomic_facts)
    print('Number of Extra Oracle:', len([d for d in mapped_prediction if d['matched_oracle_id'] is None or d['matched_oracle_id'] == 'None']))

    correct_prec = 0
    for d in mapped_prediction:    
        if d['verification'] == 1 and (d['matched_oracle_id'] is None or d['matched_oracle_id'] == 'None'):
            correct += 1
            correct_prec += 1
            total_atomic_facts += 1
        
        elif d['verification'] == 1:
            assert predict_weights[d['matched_oracle_id']] > 0
            correct += (1 * predict_weights[d['matched_oracle_id']])
            correct_prec += 1
            
    precision = correct_prec / len(mapped_prediction)
    recall = correct / total_atomic_facts

    print('Total Number of Atomic Facts:', total_atomic_facts)
    print('Precision:', correct_prec / len(mapped_prediction))
    print('Recall:', correct / total_atomic_facts)
    
    
    ### Compute relevance
    predicted_relevance = set([x['fact'] for x in predicted_atomic_facts if x['relevance'] == 1])
    mapped_prediction_relevant = [x for x in mapped_prediction if x['fact'] in predicted_relevance]
    predict_ids_relevant = defaultdict(int)
    for d in mapped_prediction_relevant:
        d['matched_oracle_id'] = mappings[d['fact']]
        if d['matched_oracle_id'] is not None and d['matched_oracle_id'] != 'None':
            predict_ids_relevant[d['matched_oracle_id']] += 1
    
    predict_weights_relevant = {k: 1/v for k, v in predict_ids_relevant.items()}
    correct_relevant = 0
    total_atomic_facts_relevant = len([x for x in oracle_atomic_facts if x['relevance'] == 1])

    correct_prec_relevant = 0
    for d in mapped_prediction_relevant:    
        if d['verification'] == 1 and (d['matched_oracle_id'] is None or d['matched_oracle_id'] == 'None'):
            correct_relevant += 1
            correct_prec_relevant += 1
            total_atomic_facts_relevant += 1
        
        elif d['verification'] == 1:
            assert predict_weights[d['matched_oracle_id']] > 0
            correct_relevant += (1 * predict_weights[d['matched_oracle_id']])
            correct_prec_relevant += 1
            
    precision_relevant = correct_prec_relevant / len(mapped_prediction_relevant)
    recall_relevant = correct_relevant / total_atomic_facts_relevant    
    
    print('Total Number of Relevant Atomic Facts:', total_atomic_facts_relevant)
    print('Precision (Relevant):', correct_prec_relevant / len(mapped_prediction_relevant))
    print('Recall (Relevant):', correct_relevant / total_atomic_facts_relevant)
    
    res = {
        "precision": precision,
        "recall": recall,
        "f1_score": 2 * precision * recall / (precision + recall + 1e-8),
        "total_atomic_facts": total_atomic_facts,
        
        "precision_relevant": precision_relevant,
        "recall_relevant": recall_relevant,
        "f1_score_relevant": 2 * precision_relevant * recall_relevant / (precision_relevant + recall_relevant + 1e-8),
        "total_atomic_facts_relevant": total_atomic_facts_relevant,
    }
    return res



def data2eval(idx_cap, save_folder):
    invalid_evals = []
    unevals = []

    for idx, cap in idx_cap:
        saved_path = os.path.join(save_folder, '{}.json'.format(idx))
        try:
            with open(saved_path, 'r') as file:
                eval = json.load(file)
            if not eval['valid']:
                invalid_evals.append([idx, cap])
        except:
            unevals.append([idx, cap])

    return unevals + invalid_evals



def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', type=str, default='iiw_data/images')
    parser.add_argument('--annotation_file', type=str, default='IIW_400_data.jsonl')
    parser.add_argument('--caption_file', type=str, required=True)
    parser.add_argument('--save_folder', type=str, required=True)
    parser.add_argument('--n_ai_eval', type=int, default=3)
    parser.add_argument('--max_n_ai_eval', type=int, default=5)
    parser.add_argument('--api_key', type=str, required=True)
    parser.add_argument('--n_parts', type=int, required=True)
    parser.add_argument('--part_idx', type=int, required=True)

    args = parser.parse_args()
    return args


def main():
    args = argparser()

    print('Initializing Client')
    client = OpenAI(
        api_key=args.api_key,
    )

    print('Initialization finished')
    
    print('Loading Captions')

    oracle_facts = load_jsonl(args.annotation_file)
    oracle_facts_map = {x["image_name"]: x for x in oracle_facts}

    annotations = load_jsonl(args.caption_file)
    
    total_size = len(annotations)
    world_size, rank = args.n_parts, args.part_idx
    shard_size = total_size // world_size
    left = total_size % world_size
    shard_sizes = [shard_size + int(r < left) for r in range(world_size)]

    begin = sum(shard_sizes[:rank])
    end = min(sum(shard_sizes[:rank + 1]), total_size)
    
    captions_part = annotations[begin:end]
    idx_part = list(range(begin, end))
    idx_captions_part = list(zip(idx_part, captions_part))
    os.makedirs(args.save_folder, exist_ok=True)
    
    idx_captions_part_to_eval = data2eval(idx_captions_part, args.save_folder)
    print('Loading finished')
    
    all_results = []

    MAX_TRIES = 5
    print('Start collecting')
    for idx, caption in tqdm(idx_captions_part_to_eval, desc="Part {}/{}".format(args.part_idx, args.n_parts)):
        for _ in range(MAX_TRIES):
            cap_dict = caption.copy()
            cap_dict["human_annotated_oracle_atomic_facts"] = [
                {"id": fact_idx, "fact": fact}
                for fact_idx, fact in enumerate(oracle_facts_map[caption['image_path']]['oracle_facts'])
            ]
            
            eval_dict = {'consume_token': []}
            
            try:
                #############################
                # Extract Atomic Facts
                pred_atomic_fact_results, completion = run_gpt_text(
                    client=client,
                    prompt=extract_prompt.format(caption=caption['caption']),
                    temperature=1,
                    n=1,
                )
                parsed_pred_atomic_facts = parse_text_json(pred_atomic_fact_results[0])
                eval_dict['pred_atomic_facts'] = parsed_pred_atomic_facts
                parsed_pred_atomic_facts_for_mapping = copy.deepcopy(parsed_pred_atomic_facts)
                for x in parsed_pred_atomic_facts_for_mapping:
                    del x['relevance']
                
                n_norm_tokens = completion.usage.prompt_tokens + 3 * completion.usage.completion_tokens
                eval_dict['consume_token'].append(n_norm_tokens)
                
                
                ########################
                # Match Prediction and Oracle (Human Annotated)
                match_results_with_human_raw, completion = run_gpt_text(
                    client=client,
                    prompt=match_prompt.format(
                        predicted_atomic_facts=format_json_input(parsed_pred_atomic_facts_for_mapping),
                        oracle_atomic_facts=format_json_input(cap_dict["human_annotated_oracle_atomic_facts"]),
                    ),
                    temperature=1,
                    n=3,
                )
                match_results_with_human = [parse_text_json(result) for result in match_results_with_human_raw]
                mappings_with_human = [{d['fact']: d['matched_oracle_id'] for d in mapping} for mapping in match_results_with_human]
                eval_dict['mappings_with_human'] = mappings_with_human
                eval_dict['mappings_with_human_raw'] = match_results_with_human_raw
                
                n_norm_tokens = completion.usage.prompt_tokens + 3 * completion.usage.completion_tokens
                eval_dict['consume_token'].append(n_norm_tokens)

                
                ########################
                # Verify Atomic Facts
                reference_caption = caption['oracle_caption']
                
                verified_results, completion = run_gpt_multimodal(
                    client=client,
                    prompt=verify_prompt.format(
                        reference_caption=reference_caption,
                        atomic_facts=format_json_input(parsed_pred_atomic_facts_for_mapping),
                    ),
                    image_path=os.path.join(args.image_path, caption['image_path']),
                    temperature=0.3,
                    detail='high',
                    n=args.n_ai_eval,
                )
                verified_results = [parse_text_json(result) for result in verified_results]
                eval_dict['verified_results'] = verified_results
                

                #######################
                # Compute Precision and Recall with human annotation
                human_annotated_oracle_atomic_facts = cap_dict["human_annotated_oracle_atomic_facts"]
                for d in human_annotated_oracle_atomic_facts:
                    d['relevance'] = 1
                final_result = {'trials': []}
                for i in range(len(verified_results)):
                    for j in range(len(mappings_with_human)):
                        res = compute_precision_recall(parsed_pred_atomic_facts, human_annotated_oracle_atomic_facts, verified_results[i], mappings_with_human[j])
                        final_result['trials'].append(res)
                        
                final_result['precision_avg'] = sum([x['precision'] for x in final_result['trials']]) / len(final_result['trials'])
                final_result['recall_avg'] = sum([x['recall'] for x in final_result['trials']]) / len(final_result['trials'])
                final_result['f1_score_avg'] = sum([x['f1_score'] for x in final_result['trials']]) / len(final_result['trials'])
                
                final_result['precision_relevant_avg'] = sum([x['precision_relevant'] for x in final_result['trials']]) / len(final_result['trials'])
                final_result['recall_relevant_avg'] = sum([x['recall_relevant'] for x in final_result['trials']]) / len(final_result['trials'])
                final_result['f1_score_relevant_avg'] = sum([x['f1_score_relevant'] for x in final_result['trials']]) / len(final_result['trials'])
                eval_dict['scores'] = final_result
                
                print('Precision:', final_result['precision_avg'])
                print('Recall:', final_result['recall_avg'])
                print('F1 Score:', final_result['f1_score_avg'])
                print('Precision (Relevant):', final_result['precision_relevant_avg'])
                print('Recall (Relevant):', final_result['recall_relevant_avg'])
                print('F1 Score (Relevant):', final_result['f1_score_relevant_avg'])
                        
                n_norm_tokens = completion.usage.prompt_tokens + 3 * completion.usage.completion_tokens
                eval_dict['consume_token'].append(n_norm_tokens)
                
                eval_dict['precision'] = final_result['precision_avg']
                eval_dict['recall'] = final_result['recall_avg']
                eval_dict['f1_score'] = final_result['f1_score_avg']
                
                eval_dict['precision_relevant'] = final_result['precision_relevant_avg']
                eval_dict['recall_relevant'] = final_result['recall_relevant_avg']
                eval_dict['f1_score_relevant'] = final_result['f1_score_relevant_avg']

                
                cap_dict['eval_details'] = eval_dict
                cap_dict['valid'] = True

                
            except Exception as e:
                traceback.print_exc()
                print(match_results_with_human_raw)
                eval_dict['error'] = str(e)
                cap_dict['eval_details'] = eval_dict
                cap_dict['valid'] = False

            all_results.append(cap_dict)
            save_file = os.path.join(args.save_folder, '{}.json'.format(idx))
            with open(save_file, 'w') as f:
                json.dump(cap_dict, f, ensure_ascii=False)
                
            if cap_dict['valid']:
                break

if __name__ == '__main__':
    main()
import os
import glob
import argparse
import json
import scipy
import numpy as np

def avg(numbers):
    return sum(numbers) / len(numbers)

parser = argparse.ArgumentParser()
parser.add_argument("--eval_folder", type=str, required=True)
args = parser.parse_args()


json_file_pattern = os.path.join(args.eval_folder, '*.json')
json_files = glob.glob(json_file_pattern)

all_results = []
for json_f in json_files:
    try:
        all_results.append([int(json_f.split('/')[-1][:-5]), json.load(open(json_f, 'r'))])
    except:
        print(json_f)

valid_results = [result for idx, result in all_results if result['valid']]
invalid_ids = [idx for idx, result in all_results if not result['valid']]

if invalid_ids:
    print('The following {} evaluations are invalid {}\nTotal Valid Evaluation: {}'.format(len(invalid_ids), invalid_ids, len(valid_results)))
else:
    print('All {} evaluations are valid'.format(len(valid_results)))

dc_score = [(result['eval_details']['f1_score_relevant'] + result['eval_details']['f1_score']) / 2 for result in valid_results]

print('DCScore: {:.2f}'.format(avg(dc_score) * 100))
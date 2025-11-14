import re
import os
import json
import datasets
from tqdm import tqdm
from verl.utils.hdfs_io import copy, makedirs
import argparse


def load_dataset(data_paths):
    def _load_dataset(data_path):        
        with open(data_path, "r", encoding="utf-8") as f:
            items = json.load(f)
        data = []
        for i, item in enumerate(items):
            d = item
            d["prompt_id"] = i
            data.append(d)
        return data
    
    data = []
    for data_path in data_paths:
        data.extend(_load_dataset(data_path))
    
    return data


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='instruction_following_optimization/verl/data/instruction_optimization')
    parser.add_argument('--hdfs_dir', default=None)

    args = parser.parse_args()
    
    data_paths = ["instruction_following_optimization/verl/data/instruction_optimization/data_examples.json"]
    data_list = load_dataset(data_paths)
    dataset = datasets.Dataset.from_list(data_list)

    # add a row to each data item that represents a unique id
    def make_map_fn(split):

        def process_fn(example, idx):
            question = example.pop('instruction')
            checklist = example.pop('checklist')
            checklist_struct = example.pop('checklist_struct')
            prompt_id = example.pop("prompt_id")
            
            data = {
                "data_source": "if",
                "prompt": [
                    {
                        "role": "user",
                        "content": question,
                    }
                ],
                "ability": "if",
                "reward_model": {
                    "style": "rm",
                    "ground_truth": json.dumps({
                        "prompt_id" : prompt_id,
                        "checklist": checklist,
                        "checklist_struct": checklist_struct
                    })
                },
                "extra_info": {
                    'split': split,
                    'index': idx,
                }
            }
            return data

        return process_fn

    train_dataset = dataset.map(function=make_map_fn('train'), with_indices=True)

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    train_dataset.to_parquet(os.path.join(local_dir, 'train.parquet'))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)

        copy(src=local_dir, dst=hdfs_dir)

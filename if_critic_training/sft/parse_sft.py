import json
import random
from critique_generation_prompts import critique_generation_prompt


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, default="sft_data_examples.json")
    parser.add_argument("--output_path", type=str, default="sft_training_data_llamafactory.json")
    args = parser.parse_args()

    with open(args.input_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    outs = []
    for d in data:
        outs.append({
            "instruction" : critique_generation_prompt.format(prompt=d["instruction"], response=d["response"], checklist=d["checklist"]),
            "input" : "",
            "output" : d["final_critique"],
            "history" : [],
        })

        
    random.seed(42)
    random.shuffle(outs)    
    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(outs, f, ensure_ascii=False, indent=4)

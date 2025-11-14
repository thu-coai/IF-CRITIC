from vllm import LLM, SamplingParams
import argparse
import json
import re
from transformers import AutoTokenizer
from checklist_generation_prompts import checklist_generation_prompt


def parse_checklist(checklist):
    pattern = re.compile(
        r'\[要求(?P<编号>\d+)-开始\]\s*'
        r'\n要求：(.*?)\s*'
        r'\n\[要求\1-结束\]',
        re.S
    )
    matches = pattern.finditer(checklist)
    results = []
    for match in matches:
        results.append({
            '编号': f"{match.group('编号')}",
            '要求': match.group(2).strip(),
        })
    if checklist.count('-开始]') != len(results) or checklist.count('-结束]') != len(results):
        return None
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="<path_to_checklist_generator>")
    parser.add_argument("--input_path", type=str, default="input_examples.json")
    parser.add_argument("--output_path", type=str, default="output_examples.json")
    parser.add_argument("--tensor_parallel_size", type=int, default=2)
    args = parser.parse_args()
    
    with open(args.input_path, "r", encoding='utf-8') as f:
        data = json.load(f)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    llm = LLM(model=args.model_path, tensor_parallel_size=args.tensor_parallel_size, dtype='bfloat16', trust_remote_code=True)

    params_dict = {
        "n": 1,
        "best_of": 1,
        "temperature": 0.0,
        "top_p": 0.9,
        "top_k": -1,
        "max_tokens": 8192,
        "skip_special_tokens": True,
        "logprobs" : True,
    }
    sampling_params = SamplingParams(**params_dict)
    prompts = []
    for d in data:
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": checklist_generation_prompt.format(prompt=d["instruction"])}
        ]
        prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
        )
        prompts.append(prompt)
    outputs = llm.generate(prompts, sampling_params)
    cnt = 0
    outs = []
    for i in range(len(data)):
        generated_text = outputs[cnt].outputs[0].text
        outs.append({
            "instruction" : data[i]["instruction"],
            "checklist" : generated_text,
            "checklist_struct" : parse_checklist(generated_text)
        })
        cnt += 1

    with open(args.output_path, "w", encoding='utf-8') as f:
        json.dump(outs, f, ensure_ascii=False, indent=4)
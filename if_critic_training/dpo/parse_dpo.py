import json
import re
import random
import numpy as np
import difflib
from copy import deepcopy
from critique_generation_prompts import critique_generation_prompt


def parse_critique(critique):
    pattern = re.compile(
        r'\[要求(?P<编号>\d+)-开始\]\s*'
        r'\n要求：(.*?)\s*'
        r'\n分析：(.*?)\s*'
        r'\n结论：(.*?)\s*'
        r'\n\[要求\1-结束\]',
        re.S
    )
    matches = pattern.finditer(critique)
    results = []
    for match in matches:
        results.append({
            '编号': f"{match.group('编号')}",
            '要求': match.group(2).strip(),
            '分析': match.group(3).strip(),
            '结论': match.group(4).strip()
        })
    for result in results:
        if result['结论'] not in ['[[人工智能助手的回复满足了该要求]]', '[[人工智能助手的回复没有满足该要求]]']:
            return None
    if critique.count('-开始]') != len(results) or critique.count('-结束]') != len(results):
        return None
    return results


def reverse_conclusion(critique):
    if "[[人工智能助手的回复满足了该要求]]" in critique["结论"]:
        critique["结论"] = "[[人工智能助手的回复没有满足该要求]]"
    else:
        critique["结论"] = "[[人工智能助手的回复满足了该要求]]"
    return critique


def parse_checklist(checklist):
    if checklist == None:
        return checklist
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


def reconstruct(critique):
    prompt = ""
    for i, c in enumerate(critique): 
        if prompt != "":
            prompt += "\n\n"
        prompt += f"[要求{i+1}-开始]\n要求：{c['要求'].strip()}\n分析：{c['分析'].strip()}\n结论：{c['结论'].strip()}\n[要求{i+1}-结束]"
    return prompt


def get_pair(right, wrong):
    r, w = 0, 0
    if "[[人工智能助手的回复满足了该要求]]" in right["结论"]:
        r = 1
    if "[[人工智能助手的回复满足了该要求]]" in wrong["结论"]:
        w = 1
    return (r, w)


def mbr_select(critiques):
    sim = []
    max_sim = -1e9
    right_unit = None
    for i in range(len(critiques)):
        sim.append(0)
        for j in range(len(critiques)):
            sim[i] += difflib.SequenceMatcher(None, critiques[i]["分析"], critiques[j]["分析"]).quick_ratio()
        if sim[i] > max_sim:
            max_sim = sim[i]
            right_unit = critiques[i]
    return right_unit
    
    
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--num", type=int, default=1)
    parser.add_argument("--input_path", type=str, default="dpo_data_examples.json")
    parser.add_argument("--output_path", type=str, default="dpo_training_data_llamafactory.json")
    args = parser.parse_args()

    with open(args.input_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    random.seed(42)
    np.random.seed(42)
    
    dpo_data = []
    perference_dis = {}

    for d in data:
        parsed_critique = parse_critique(d["final_critique"])
        parsed_checklist = parse_checklist(d["checklist"])
        if parsed_critique == None or parsed_checklist == None or len(parsed_critique) != len(parsed_checklist):
            continue
        critiques = []
        pair_data = []
        right_critiques = [[] for _ in range(len(parsed_checklist))]

        for i in range(10):
            critique = parse_critique(d[f"generated_critique_{i}"])
            if critique == None or len(critique) != len(parsed_critique):
                continue
            flag = True
            for id, (u, v) in enumerate(zip(critique, parsed_critique)):
                if u["要求"] != v["要求"]:
                    flag = False
            if flag == False:
                continue
            for id, (u, v) in enumerate(zip(critique, parsed_critique)):
                if u["结论"] == v["结论"]:
                    right_critiques[id].append(u)    
            critiques.append(critique)


        critique_candidates = []
        for c_id, critique in enumerate(critiques):
            flag = 0
            for u_id, (u, v) in enumerate(zip(critique, parsed_critique)):
                if u["结论"] != v["结论"]:
                    flag = 1
                    if len(right_critiques[u_id]) == 0:
                        flag = -1
                        break
            if flag == 1:
                critique_candidates.append(critique)
        
        if len(critique_candidates) > args.num:
            critique_candidates = random.sample(critique_candidates, args.num)
        
        for critique in critique_candidates:
            rejected_critique = deepcopy(critique)
            chosen_critique = []
            for u_id, (u, v) in enumerate(zip(critique, parsed_critique)):
                if u["结论"] != v["结论"]:
                    right_unit = mbr_select(right_critiques[u_id])
                    chosen_critique.append(deepcopy(right_unit))
                    perference_dis[get_pair(u, right_unit)] = perference_dis.get(get_pair(u, right_unit), 0) + 1
                else:
                    chosen_critique.append(deepcopy(u))
            
            dpo_data.append({
                "instruction" : critique_generation_prompt.format(prompt=d["instruction"], response=d["response"], checklist=d["checklist"]),
                "input" : "",
                "chosen" : reconstruct(chosen_critique),
                "rejected" : reconstruct(rejected_critique),
                "history" : []
            })

    print(len(dpo_data))
    print(perference_dis)
    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(dpo_data, f, ensure_ascii=False, indent=4)
import os
import json
from tqdm import tqdm
from random import sample

from llm import LLM
from config.config import LLMConfig


colors = ["red", "green", "blue", "orange"]

TP = LLM(LLMConfig("TP_OL", "Cubes"))
OD = LLM(LLMConfig("OD", "Cubes"))

def get_instruction(query:str):
    instruction = f"objects = {['blue_cube', 'green_cube', 'orange_cube', 'red_cube']}\n"
    instruction += f"# Query: {query}"
    return instruction

tasks = ['stack', 'L', 'pyramid']

for i in range(5):
    queries = [
        "make a stack of cubes on top of the {} cube".format(*sample(colors, 1)),
        "rearrange cubes to write the letter L flat on the table. keep {} at its location".format(*sample(colors, 1)),
        "build a pyramid with the {} and {} cubes at the base and {} cube at the top. keep {} cube at its original position.".format(*(2*sample(colors, 3)))
    ]

    for j, t in enumerate(tasks):
        query = queries[j]
        plan = TP.run(get_instruction(query), short_history=True)
        optimizations = []
        for q in tqdm(plan['tasks']):
            if q not in ['open_gripper()', 'close_gripper()']:
                try:
                    opt = OD.run(get_instruction(q), short_history=True)
                    if "instruction" not in opt.keys():
                        optimizations.append(opt)
                    else:
                        optimizations.append(None)
                except:
                    optimizations.append(None)
            else:
                optimizations.append(None)

        data = {"query": query, "plan": plan, "optimizations": optimizations}
        data_folder = f"data/llm_responses/{t}"
        n_files = len(os.listdir(data_folder))
        json.dump(data, open(f"{data_folder}/{n_files}.json", "w"), indent=4)

import os
import json
from tqdm import tqdm
from random import sample

from llm import LLM
from config.config import LLMConfig


colors = ["red", "green", "blue", "orange"]

def get_instruction(query:str, task:str):
    if task in ['stack', 'L', 'pyramid']:
        instruction = f"objects = {['blue_cube', 'green_cube', 'orange_cube', 'red_cube']}\n" 
    elif task == 'CleanPlate':
        instruction = f"objects = {['plate', 'sponge']}\n" 
    elif task == 'Sponge':
        instruction = f"objects = {['sponge', 'container', 'container_handle', 'sink']}\n"
    instruction += f"# Query: {query}"
    return instruction

method = 'ours_objective'
tasks = ['Sponge']

TP = LLM(LLMConfig("TP", "Sponge"))
OD = LLM(LLMConfig("OD", "Sponge"))

for i in range(48):
    queries = [
        #"make a stack of cubes on top of the {} cube".format(*sample(colors, 1)),
        #"rearrange cubes to write the letter L flat on the table. keep {} at its location".format(*sample(colors, 1)),
        #"build a pyramid with the {} and {} cubes at the base and {} cube at the top. keep {} cube at its original position.".format(*(2*sample(colors, 3)))
        #"clean the plate with the sponge. (go above the plate before starting cleaning)",
        "use right robot to move container to sink and left robot to move sponge to the sink. the sponge is wet so keep it above the container to avoid water dropping on the floor"
    ]

    for j, t in enumerate(tasks):
        query = queries[j]
        plan = TP.run(get_instruction(query, t), short_history=True)
        optimizations = []
        for q in tqdm(plan['tasks']):
            if q not in ['open_gripper()', 'close_gripper()']:
                try:
                    opt = OD.run(get_instruction(q, t), short_history=True)
                    if "instruction" not in opt.keys():
                        optimizations.append(opt)
                    else:
                        optimizations.append(None)
                except:
                    optimizations.append(None)
            else:
                optimizations.append(None)
        #if t =="pyramid": query = query[:-49]
        data = {"query": query, "plan": plan, "optimizations": optimizations}
        data_folder = f"data/{method}/llm_responses/{t}"
        n_files = len(os.listdir(data_folder))
        json.dump(data, open(f"{data_folder}/{n_files}.json", "w"), indent=4)

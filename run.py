import os
import re
import json
from tqdm import tqdm

from simulation_local import Simulation

def extract_number(filename):
    match = re.search(r'(\d+)', filename)
    if match:
        return int(match.group(0))
    return 0


if __name__ == "__main__":

    s = Simulation()
    method = "ours" # in ['ours', 'ours_objective']
    tasks = ["CookSteak"]#"stack", "L", "pyramid"] # 'stack', 'pyramid', 'L', 'CleanPlate'
    
    for t in tasks:
        print("\nRunning task: ", t)
        task_folder = f'data/{method}/llm_responses/{t}'
        for file in tqdm(sorted(os.listdir(task_folder), key=extract_number)):
            # reset env
            s.reset()
            # load data
            data = json.load(open(f"{task_folder}/{file}", 'r'))
            if "objective" in method:
                optimizations = [{"objective":o["objective"], "equality_constraints":[], "inequality_constraints":[]} if o is not None else None for o in data["optimizations"]]
            else:
                optimizations = data['optimizations']
            # run sim
            s.run(data["query"] + f'(file: {file})', data["plan"], optimizations)

    s.close()

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
    tasks = ["stack", "L", "pyramid"]
    
    for t in tasks:
        print("\nRunning task: ", t)
        task_folder = f'data/llm_responses/{t}'
        for file in tqdm(sorted(os.listdir(task_folder), key=extract_number)[46:51]):
            # reset env
            s.reset()
            # load data
            data = json.load(open(f"{task_folder}/{file}", 'r'))
            # run sim
            s.run(data["query"] + f'(file: {file})', data["plan"], data["optimizations"])

    s.close()

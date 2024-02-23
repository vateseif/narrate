import os
import json

from simulation_local import Simulation


if __name__ == "__main__":

    s = Simulation()
    tasks = ["stack", "L"]
    
    for t in tasks:
        print("\nRunning task: ", t)
        task_folder = f'data/llm_responses/{t}'
        for file in sorted(os.listdir(task_folder)):
            print(f"File: {file}")
            # reset env
            s.reset()
            # load data
            data = json.load(open(f"{task_folder}/{file}", 'r'))
            # run sim
            s.run(data["query"] + f'(file: {file})', data["plan"], data["optimizations"])

    s.close()

import requests
from tqdm import tqdm
from random import sample
import subprocess
import time

base_url = 'http://localhost:8080/'

task_names = ["stacking", "letter_l", "pyramid"]
colors = ["red", "green", "blue", "orange"]
prompts = ["make a stack of cubes on top of the {} one".format(*sample(colors, 1)),
           "rearrange cubes to write the letter L on the table. keep {} at its location".format(*sample(colors, 1)),
           "build a pyramid with the {} and {} cubes at the base and {} cube at the top. keep {} cube at its original position.".format(*(2*sample(colors, 3)))]

N_EXPERIMENTS = 50
for prompt, task_name in zip(prompts, task_names):
    process = subprocess.Popen(['python', 'simulation.py', '--task_name', task_name])
    time.sleep(10)

    for i in tqdm(range(N_EXPERIMENTS)):
        try:
            response = requests.post(base_url+'cap', json={"content": prompt, "task_name": task_name})
        except Exception as e:
            print(e)
        requests.get(base_url+'reset')
    
    process.kill()
    time.sleep(15)

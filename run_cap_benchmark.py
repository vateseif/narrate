import requests
from tqdm import tqdm
from random import sample
base_url = 'http://localhost:8080/'

colors = ["red", "green", "blue", "orange"]
prompts = ["make a stack of cubes on top of the {} one".format(*sample(colors, 1)),
           "rearrange cubes to write the letter L on the table. keep {} at its location".format(*sample(colors, 1)),
           "build a pyramid with the {} and {} cubes at the base and {} cube at the top. keep {} cube at its original position.".format(*(2*sample(colors, 3)))]

N_EXPERIMENTS = 1
prompt = prompts[0]

for i in tqdm(range(N_EXPERIMENTS)):
    response = requests.post(base_url+'cap', json={"content": prompt})
    requests.get(base_url+'reset')
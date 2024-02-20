import requests
from tqdm import tqdm
base_url = 'http://localhost:8080/'

N_EXPERIMENTS = 50
prompt = "stack all the cubes in a single pile."

for i in tqdm(range(N_EXPERIMENTS)):
    response = requests.post(base_url+'cap', json={"content": prompt})
    requests.get(base_url+'reset')
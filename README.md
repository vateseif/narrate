# NARRATE: Versatile Language Architeture for Optimal Control in Robotics
This repo contains a reference implementation of the paper [NARRATE](https://narrate-mpc.github.io). 
We refer the reader to our project page to find the paper and details about the method.


https://github.com/vateseif/l2o/assets/45405956/4c9b84e3-fd7d-4159-903d-232852029f7e


## Setup
### Env
Create a python environment (i.e. with conda):
~~~
conda create --name narrate python=3.9
conda activate narrate
~~~
### Requirements
Install requirements
~~~
pip install -r requirements.txt
~~~
### OpenAI key
You need to create the file `keys/gpt4.key` and put your OpenAI key. Make sure to have acces to GPT4. 

## Run
You will need to run 2 files in order to interact with the simulation environment.

To start the chat interface you have to execute in your terminal:
~~~
streamlit run main.py
~~~

To start the simulation you have to execute in your terminal
~~~
python simulation_http.py
~~~

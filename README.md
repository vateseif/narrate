# Language to Optimization
This section focuses on how to convert an instruction in English to reward and constraints of an optimization-based controller (Model Predictive Control)

## Setup
### Env
Create the conda env
~~~
create --name l2o python=3.9
conda activate l2o
~~~
### Requirements
While being in this folder (`l2o/`):
~~~
pip install -r requirements.txt
~~~
In the requirements, a custom version of the Safe-Panda-Gym is added. In case that is changed, you should uninstall and reinstall it:
~~~
pip uninstall panda_gym
pip install -r requirements.txt
~~~
### OpenAI key
You need to create the file `keys/gpt4.key` and put your OpenAI key. Make sure to have acces to GPT4. 

## Run
~~~
streamlit run main.py
~~~
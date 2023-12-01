# Language to Optimization
This section focuses on how to convert an instruction in English to reward and constraints of an optimization-based controller (Model Predictive Control)


https://github.com/vateseif/l2o/assets/45405956/4c9b84e3-fd7d-4159-903d-232852029f7e


## Setup
### Env
Create the conda env
~~~
conda create --name l2o python=3.9
conda activate l2o
~~~
### Requirements
Install requirements
~~~
pip install -r requirements.txt
~~~
### OpenAI key
You need to create the file `keys/gpt4.key` and put your OpenAI key. Make sure to have acces to GPT4. 

## Run
~~~
streamlit run main.py
~~~

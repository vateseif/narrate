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

## Run
You can run the code in an interactive way as follows:
~~~
python -i main.py
~~~
You will be able to give a task to the robot from terminal:
~~~
sim.create_plan("Stack all cubes on top of cube_2")
~~~
You can then trigger each task consecutively by running:
~~~
sim.next_task()
~~~


If you want to give feedback to the Optimization Designer (OD), you can run:
~~~
sim._solve_task("cube_2 has fallen down, you should go pick it up again")
~~~
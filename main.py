# NOTE: use conda activate safepanda for this env
import gym
import threading
import panda_gym
import numpy as np
from time import sleep
from typing import Tuple

from llm import Plan
from robot import BaseRobot
from mocks.mocks import nmpcMockOptions

class Sim:
  def __init__(self) -> None:
    # params
    self.n_episodes = 100
    # robot
    self.robot = BaseRobot()
    # env
    self.env = gym.make("PandaCubes-v2", render=True)
    self.observation = self.env.reset()

  def reset(self):
    # reset pand env
    observation = self.env.reset()
    # store observation
    self.x0 = observation[0:3]
    # reset controller
    self.robot.reset(self.x0)
    # count number of tasks solved from a plan 
    self.task_counter = 0

  def create_plan(self, user_task:str, solve=False): 
    self.plan: Plan = self.robot.create_plan(user_task)
    #self.plan: Plan = nmpcMockOptions[self.robot.cfg.controller_type]
    #sleep(5)
    print(f"\33[92m {self.plan.tasks} \033[0m \n")
    if solve:
      for _ in self.plan.tasks:
        self.next_task()
        sleep(3)


  def step(self, action: np.ndarray):
    self.observation, _, done, _ = self.env.step(action)
    # store observation
    self.x0 = self.observation[0:3]

    self.x_cube1 = self.observation[-15:-12]
    self.x_cube2 = self.observation[-18:-15]
    self.x_cube3 = self.observation[-21:-18]
    self.x_cube4 = self.observation[-24:-21]
    return done

  def get_x_cubes(self) -> Tuple[np.ndarray]:
    return (self.x_cube1, self.x_cube2, self.x_cube3, self.x_cube4)

  def _solve_task(self, plan:str):
    self.robot.next_plan(plan, self.get_x_cubes())
    return

  def next_task(self):
    self._solve_task(self.plan.tasks[self.task_counter])
    self.task_counter += 1

  def run(self):
    for _ in range(self.n_episodes):
      self.reset()
      while True:
        # update controller
        self.robot.set_x0(self.x0)
        # compute action
        action = self.robot.step()
        # step env
        done = self.step(action)
        
        sleep(0.1)
        if done:
            break

    self.env.close()

if __name__ == "__main__":

  # simulator
  sim = Sim()

  thread = threading.Thread(target=sim.run)
  thread.daemon = True  # Set the thread as a daemon (will exit when the main program ends)
  thread.start()

  #sim.create_plan("Stack all cubes on top of cube_2.")
  #sim.next_task()
  sleep(3)

  """
  optimization = Optimization(
    objective= "ca.norm_2(x - cube_4)**2",
    constraints= [
      "0.046 - ca.norm_2(x - cube_1)", 
      "0.046 - ca.norm_2(x - cube_2)", 
      "0.046 - ca.norm_2(x - cube_3)", 
      "0.046 - ca.norm_2(x - cube_4)"
    ]
  )
  sim.robot.MPC.apply_gpt_message(optimization, sim.get_x_cubes())

  sleep(10)

  optimization = Optimization(
    objective= "ca.norm_2(x - cube_1)**2",
    constraints= [
      "0.046 - ca.norm_2(x - cube_1)", 
      "0.046 - ca.norm_2(x - cube_2)", 
      "0.046 - ca.norm_2(x - cube_3)",
      "0.046 - ca.norm_2(x - cube_4)"
    ]
  )
  sim.robot.MPC.apply_gpt_message(optimization, sim.get_x_cubes())

  sleep(20)
#
  optimization = Optimization(
    objective= "ca.norm_2(x - cube_4)**2",
    constraints= [
      "0.046 - ca.norm_2(x - cube_1)", 
      "0.046 - ca.norm_2(x - cube_2)", 
      "0.046 - ca.norm_2(x - cube_3)", 
      "0.046 - ca.norm_2(x - cube_4)"
    ]
  )
  sim.robot.MPC.apply_gpt_message(optimization, sim.get_x_cubes())
  """
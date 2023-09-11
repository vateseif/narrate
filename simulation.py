import threading
from time import sleep
from typing import Optional

from llm import Plan
from robot import BaseRobot
from core import AbstractSimulation
from config.config import SimulationConfig
from mocks.mocks import nmpcMockOptions # TODO


class Simulation(AbstractSimulation):
  def __init__(self, cfg=SimulationConfig()) -> None:
    super().__init__(cfg)

    # TODO: account for multiple robots
    self.robot = BaseRobot()
    # count number of tasks solved from a plan 
    self.task_counter = 0

  def reset(self):
    # reset pand env
    self.observation = self.env.reset()
    # reset controller
    self.robot.reset(self.observation["robot_0"][:3])
    # count number of tasks solved from a plan 
    self.task_counter = 0

  def create_plan(self, user_task:str, wait_s:Optional[int]=None): 
    self.plan = self.robot.create_plan(user_task) if self.cfg.mock_plan is None else nmpcMockOptions[self.cfg.mock_plan]
    print(f"\33[92m {self.plan.tasks} \033[0m \n")
    if wait_s is not None:
      for _ in self.plan.tasks:
        self.next_task()
        sleep(wait_s)

  def step(self):
    # update controller (i.e. set the current gripper position)
    self.robot.set_x0(self.observation["robot_0"][:3])
    # compute action
    action = [self.robot.step()] # TODO: this is a list because the env may have multiple robots
    # apply action
    self.observation, _, done, _ = self.env.step(action)

    return done

  def _solve_task(self, plan:str):
    self.robot.next_plan(plan, self.observation)
    return

  def next_task(self):
    self._solve_task(self.plan.tasks[self.task_counter])
    self.task_counter += 1

  def _run(self):
    self.reset()
    while True:
      # step env
      done = self.step()
      sleep(0.1)
      if done:
          break

    self.env.close()

  def run(self):
    """ Executes self._run() in a separate thread"""
    thread = threading.Thread(target=self._run)
    thread.daemon = True  # Set the thread as a daemon (will exit when the main program ends)
    thread.start()
    return
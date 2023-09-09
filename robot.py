import numpy as np
from typing import Tuple
from llm import BaseLLM
from core import AbstractRobot
from config.config import BaseRobotConfig, BaseLLMConfigs
from controller import BaseController, ControllerOptions



class BaseRobot(AbstractRobot):

  def __init__(self, cfg=BaseRobotConfig()) -> None:
    self.cfg = cfg

    self.gripper = 1. # 1 means the gripper is open
    self.gripper_timer = 0
    self.TP = BaseLLM(BaseLLMConfigs[self.cfg.tp_type]())
    self.OD = BaseLLM(BaseLLMConfigs[self.cfg.od_type]())
    self.MPC: BaseController = ControllerOptions[self.cfg.controller_type]()

  def open_gripper(self):
    self.gripper = 0.
    self.gripper_timer = 0

  def close_gripper(self):
    self.gripper = -1.

  def set_x0(self, x0: np.ndarray):
    self.MPC.set_x0(x0)
    return

  def create_plan(self, user_task:str):
    plan = self.TP.run(user_task)
    return plan # TODO: plan.tasks is hardcoded here

  def next_plan(self, plan:str, x_cubes: Tuple[np.ndarray]):
    # print plan
    print(f"\033[91m Task: {plan} \033[0m ")
    # if custom function is called apply that
    if "open" in plan.lower() and "gripper" in plan.lower():
      self.open_gripper()
      print(f"\33[92m open_gripper() \033[0m \n")
      return
    elif "close" in plan.lower() and "gripper" in plan.lower():
      self.close_gripper()
      print(f"\33[92m close_gripper() \033[0m \n")
      return
    # design optimization functions
    optimization = self.OD.run(plan)
    # apply optimization functions to MPC
    self.MPC.apply_gpt_message(optimization, x_cubes)
    return


  def step(self):
    action = np.hstack((self.MPC.step(), self.gripper))  
    if self.gripper==0 and self.gripper_timer>self.cfg.open_gripper_time: 
      self.gripper = 1.
    else:
      self.gripper_timer += 1 
    return action
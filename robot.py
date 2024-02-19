import numpy as np
from time import time, sleep

from llm import LLM
from core import AbstractRobot
from controller import Controller
from typing import Tuple, List, Dict
from config.config import RobotConfig, LLMConfig, ODConfig


class Robot(AbstractRobot):
  def __init__(self, env_info:Tuple[List], cfg=RobotConfig()) -> None:
    self.cfg = cfg

    self.gripper = 1. # 1 means the gripper is open
    self.gripper_timer = 0
    self.TP = LLM(LLMConfig("TP_OL", self.cfg.task))
    self.OD = LLM(LLMConfig("OD", self.cfg.task))
    
    self.MPC = Controller(env_info)

  def init_states(self, observation:Dict[str, np.ndarray], t:float):
      """ Update simulation time and current state of MPC controller"""
      self.MPC.init_states(observation, t, self.gripper==-1.)

  def pretty_print(self, response:dict):
    if "instruction" in response.keys():
      pretty_msg = "**Reasoning:**\n"
      pretty_msg += f"{response['reasoning']}\n"
      pretty_msg += "**Instruction:**\n"
      pretty_msg += f"{response['instruction']}\n"
    else:
      pretty_msg = "```\n"
      pretty_msg += f"min {response['objective']}\n"
      pretty_msg += f"s.t.\n"
      for c in response['equality_constraints']:
        pretty_msg += f"\t {c} = 0\n"
      for c in response['inequality_constraints']:
        pretty_msg += f"\t {c} <= 0\n"
      pretty_msg += "```\n"
    
    return pretty_msg

  def _open_gripper(self):
    self.gripper = 0.
    self.gripper_timer = 0

  def _close_gripper(self):
    self.gripper = -1.

  def plan_task(self, user_message:str, base64_image=None) -> str:
    """ Runs the Task Planner by passing the user message and the current frame """
    return self.TP.run(user_message, base64_image, short_history=True)

  def solve_task(self, plan:str, attempt:int=0) -> str:
    """ Applies and returns the optimization designed by the Optimization Designer """
    # if custom function is called apply that
    if attempt>1:
      print("ERROR: Too many attempts.")
      return "ERROR"
    
    print(f"{self.MPC.prev_cost - self.MPC.cost} <= {self.cfg.COST_DIIFF_THRESHOLD} or {self.MPC.cost} <= {self.cfg.COST_THRESHOLD}")
    if self.MPC.prev_cost - self.MPC.cost <= self.cfg.COST_DIIFF_THRESHOLD or self.MPC.cost <= self.cfg.COST_THRESHOLD or time()-self.t_prev_task>=self.cfg.TIME_THRESHOLD:
      print("SWITCH.")
    else:
      return None
    
    self.t_prev_task = time()

    if "open_gripper" in plan.lower():
      self._open_gripper()
      #simulate_stream("OD", "\n```\n open_gripper()\n```\n")
      return "\n```\n open_gripper()\n```\n"
    elif "close_gripper" in plan.lower():
      self._close_gripper()
      #simulate_stream("OD", "\n```\n close_gripper()\n```\n")
      return "\n```\n close_gripper()\n```\n"
    # catch if reply cannot be parsed. i.e. when askin the LLM a question
    try:
      # design optimization functions
      optimization = self.OD.run(plan)
      # apply optimization functions to MPC
      self.MPC.setup_controller(optimization)
      return self.pretty_print(optimization)
    except Exception as e:
      print(f"Error: {e}")
      self.solve_task('There was an error in the formulation. Please retry with a simpler formulation.', attempt+1)

  def step(self):
    action = []
    # compute actions from controller (single or dual robot)
    control: List[np.ndarray] = self.MPC.step()
    for u in control:
      action.append(np.hstack((u, self.gripper)))  
    
    # Logic for opening and closing gripper
    if self.gripper==0 and self.gripper_timer>self.cfg.open_gripper_time: 
      self.gripper = 1.
    else:
      self.gripper_timer += 1

    return action

  def retrieve_trajectory(self):
    return self.MPC.retrieve_trajectory()
    
    
    
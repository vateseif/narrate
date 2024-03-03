import numpy as np
from time import time

from llm import LLM
from core import AbstractRobot
from controller import Controller
from typing import Tuple, List, Dict, Optional
from config.config import RobotConfig, LLMConfig, ControllerConfig


class Robot(AbstractRobot):
	def __init__(self, env_info:Tuple[List], cfg=RobotConfig()) -> None:
		self.cfg = cfg
		self.robots_info, self.objects_info = env_info

		self.gripper = 1. # 1 means the gripper is open
		self.gripper_timer = 0
		self.gripper_is_moving = False
		self.t_prev_task = time()
		self.TP = LLM(LLMConfig("TP", self.cfg.task))
		self.OD = LLM(LLMConfig("OD", self.cfg.task))
		
		self.MPC = Controller(env_info, ControllerConfig(self.cfg.task))

	def init_states(self, observation:Dict[str, np.ndarray], t:float):
		""" Update simulation time and current state of MPC controller"""
		self.MPC.init_states(observation, t, self.gripper==-0.02)

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
	
	def _get_instruction(self, query:str) -> str:
		instruction = f"objects = {[o['name'] for o in self.objects_info]}\n"
		instruction += f"# Query: {query}"
		return instruction

	def _open_gripper(self):
		self.gripper = -0.01
		self.gripper_timer = 0
		self.gripper_is_moving = True

	def _close_gripper(self):
		self.gripper = -0.02
		self.gripper_timer = 0
		self.gripper_is_moving = True

	def reset(self):
		# open grfipper
		self.gripper = 1.
		# reset llms
		self.TP.reset()
		self.OD.reset()
		# reset mpc
		self.MPC.reset()

	def plan_task(self, user_message:str, base64_image=None) -> str:
		""" Runs the Task Planner by passing the user message and the current frame """
		plan = self.TP.run(self._get_instruction(user_message), base64_image, short_history=True)
		print(f"\33[92m {plan} \033[0m \n")
		return plan
	
	def is_robot_busy(self):
		return not ((np.abs(self.MPC.prev_cost - self.MPC.cost) <= self.cfg.COST_DIIFF_THRESHOLD or 
			  		self.MPC.cost <= self.cfg.COST_THRESHOLD or
			  		time()-self.t_prev_task>=self.cfg.TIME_THRESHOLD) and 
					not self.gripper_is_moving)

	def update_gripper(self, plan:str) -> Optional[str]:
		if "open_gripper" in plan.lower():
			self._open_gripper()
			#simulate_stream("OD", "\n```\n open_gripper()\n```\n")
			return "\n```\n open_gripper()\n```\n"
		elif "close_gripper" in plan.lower():
			self._close_gripper()
			#simulate_stream("OD", "\n```\n close_gripper()\n```\n")
			return "\n```\n close_gripper()\n```\n"
		else:
			return None

	def solve_task(self, query:str, optimization:Optional[dict]=None) -> str:
		""" Applies and returns the optimization designed by the Optimization Designer """
		# if custom function is called apply that

		if self.is_robot_busy(): 
			return None
				
		self.t_prev_task = time()

		gripper_update = self.update_gripper(query)
		if gripper_update is not None:
			return gripper_update

		if optimization is not None:
			# apply optimization functions to MPC
			try:
				self.MPC.setup_controller(optimization)
				return self.pretty_print(optimization)
			except Exception as e:
				print(f"Error with Open Loop Optimization: {e}")

		# catch if reply cannot be parsed. i.e. when askin the LLM a question
		for i in range(self.cfg.MAX_OD_ATTEMPTS):
			try:
				# design optimization functions
				if i == 0:
					query += "The previous optimization was not feasible. Please try again with a simpler formulation. You can assume the size of all objects is the same."
				optimization = self.OD.run(self._get_instruction(query), short_history=True)
				print(f"\33[92m {optimization} \033[0m \n")
				if self.cfg.method == "objective":
					optimization["equality_constraints"] = []
					optimization["inequality_constraints"] = []
				# apply optimization functions to MPC
				self.MPC.setup_controller(optimization)
				return self.pretty_print(optimization)
			except Exception as e:
				print(f"Error: {e}")

		return None

	def step(self):
		action = []
		# compute actions from controller (single or dual robot)
		control: List[np.ndarray] = self.MPC.step()
		for u in control:
			action.append(np.hstack((u, self.gripper)))  
			
		if self.gripper_timer >= self.cfg.open_gripper_time:
			self.gripper_is_moving = False
		else:
			self.gripper_timer += 1
			self.gripper_is_moving = True
			if self.gripper_timer == int(self.cfg.open_gripper_time/2) and self.gripper == -0.01:
				self.gripper = 1.

		return action

	def retrieve_trajectory(self):
		return self.MPC.retrieve_trajectory()
		
		
		
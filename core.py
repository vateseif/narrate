import os
import inspect
from typing import List
from abc import abstractmethod

import gym
import panda_gym
import numpy as np
from langchain.chat_models import ChatOpenAI

# GPT4 api key
os.environ["OPENAI_API_KEY"] = open(os.path.dirname(__file__) + '/keys/gpt4.key', 'r').readline().rstrip()


class AbstractLLMConfig:
  prompt: str
  parsing: str
  model_name: str
  temperature: float

class AbstractControllerConfig:
  T: int 
  nx: int  
  nu: int  
  dt: float 
  lu: float # lower bound on u
  hu: float # higher bound on u

class AbstractRobotConfig:
  name: str
  controller_type: str

class AbstractSimulaitonConfig:
  env_name: str
  render: bool

class ObjBase:
  '''
  The object base that defines debugging tools
  '''
  def initialize (self, **kwargs):
    pass

  def sanityCheck (self):
    # check the system parameters are coherent
    return True

  def errorMessage (self,msg):
    print(self.__class__.__name__+'-'+inspect.stack()[1][3]+': [ERROR] '+msg+'\n')
    return False

  def warningMessage (self,msg):
    print(self.__class__.__name__+'-'+inspect.stack()[1][3]+': [WARNING] '+msg+'\n')
    return False



class AbstractController(ObjBase):

  def __init__(self, cfg: AbstractControllerConfig) -> None:
    self.cfg = cfg
    
  @abstractmethod
  def reset(self, x0:np.ndarray) -> None:
    return

  @abstractmethod
  def apply_gpt_message(self, gpt_message:str) -> None:
    return
  
  @abstractmethod
  def step(self) -> np.ndarray:
    return


class AbstractLLM(ObjBase):

  def __init__(self, cfg:AbstractLLMConfig) -> None:
    self.cfg = cfg
    # init model
    self.model = ChatOpenAI(model_name=self.cfg.model_name, temperature=self.cfg.temperature)


  @abstractmethod
  def run(self):
    return



class AbstractRobot(ObjBase):

  def __init__(self, cfg:AbstractRobotConfig) -> None:
    self.cfg = cfg

    # components
    self.TP: AbstractLLM          # Task planner
    self.OD: AbstractLLM          # Optimization Designer
    self.MPC: AbstractController  # Controller


  @abstractmethod
  def reset_gpt(self):
    return
  
  def reset_controller(self, x0:np.ndarray):
    self.MPC.reset(x0)
    return
    
  def reset(self, x0:np.ndarray):
    self.reset_gpt()
    self.reset_controller(x0)
    return


class AbstractSimulation(ObjBase):
  def __init__(self, cfg: AbstractSimulaitonConfig) -> None:
    self.cfg = cfg
    # init robots
    self.robot: AbstractRobot # TODO: account for multiple robots
    # init env
    self.env = gym.make(f"Panda{cfg.env_name}-v2", render=cfg.render)
    # count number of tasks solved from a plan 
    self.task_counter = 0

  def reset(self):
    """ Reset environment """
    pass

  def create_plan(self):
    """ Triggers the Task Planner to generate a plan of subtasks"""
    pass

  def next_task(self):
    """ Tasks the Optimization Designer to carry out the next task in the plam"""
    pass

  def _solve_task(self):
    """ Applies the optimization designed by the Optimization Designer"""
    pass    

  def _run(self):
    """ Start the simulation """
    pass

  def run(self):
    """ Executes self._run() in a separate thread"""
    pass
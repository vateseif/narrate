import os
import inspect
import numpy as np
from abc import abstractmethod
from langchain.chat_models import ChatOpenAI

# GPT4 api key
os.environ["OPENAI_API_KEY"] = open(os.path.dirname(__file__) + '/keys/gpt_seamless.key', 'r').readline().rstrip()


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
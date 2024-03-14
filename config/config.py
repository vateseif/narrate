from prompts.prompts import PROMPTS
from core import AbstractControllerConfig, AbstractLLMConfig, AbstractRobotConfig, AbstractSimulaitonConfig
from typing import List


class SimulationConfig(AbstractSimulaitonConfig):
  render: bool = True
  debug: bool = False
  logging: bool = False
  logging_video: bool = False
  logging_video_fps: int = 5
  logging_video_frequency: int = 10
  task: str = "Cubes"     # [Cubes, CleanPlate, Sponge, CookSteak]
  save_video: bool = False
  fps: int = 20 # only used if save_video = True
  dt: float = 0.05 # simulation timestep. Must be equal to that of controller
  frame_width: int = 1024
  frame_height: int = 1024
  frame_target_position: List[float] = [0.0, -0.1, 0.]
  frame_distance: float = 1.6
  frame_yaw: int = -125
  frame_pitch: int = -30
  method:str = 'ours'


class LLMConfig(AbstractLLMConfig):
  def __init__(self, avatar:str, task:str=None) -> None:
    self.avatar: str = avatar
    self.mock_task = None # TODO wtf this is shit
    self.prompt: str = PROMPTS[avatar][task] # TODO: this is bad. Only works for Optimization now
  model_name: str = "gpt-4-0125-preview"
  streaming: bool = False
  temperature: float = 0.9
  max_tokens: int = 1000


class ControllerConfig(AbstractControllerConfig):
  def __init__(self, task:str=None) -> None:
    self.task: str = task
  nx: int = 3
  nu: int = 3 
  T: int = 15
  dt: float = 0.05
  lu: float = -0.2 # lower bound on u
  hu: float = 0.2  # higher bound on u
  model_type: str = "discrete"
  penalty_term_cons: float = 1e7
  

class RobotConfig(AbstractRobotConfig):
  def __init__(self, task:str=None) -> None:
    self.task: str = task
  open_gripper_time: int = 28
  method: str = "optimization" # ['optimization', 'objective']
  COST_THRESHOLD: float = 1e-5
  COST_DIIFF_THRESHOLD: float = 1e-7
  GRIPPER_WIDTH_THRESHOLD: float = 4e-6
  TIME_THRESHOLD: float = 25
  MAX_OD_ATTEMPTS: int = 2


class DBConfig:
  db_name: str = "data/DBs/cubes_objective.db"



from core import AbstractControllerConfig, AbstractLLMConfig, AbstractRobotConfig, AbstractSimulaitonConfig
from prompts.prompts import *


class SimulationConfig(AbstractSimulaitonConfig):
  render: bool = True
  debug: bool = True
  env_name: str = "Cubes"     # [Cubes, CleanPlate, Sponge, MoveTable]
  task: str = "stack"  # [None, "stack", "pyramid", "L", "reverse", "clean_plate", "sponge", "move_table"]
  save_video: bool = False
  fps: int = 20 # only used if save_video = True
  dt: float = 0.05 # simulation timestep. Must be equal to that of controller
  width: int = 1024
  height: int = 1024


class ObjectivePlanLLMConfig(AbstractLLMConfig):
  prompt: str = OBJECTIVE_TASK_PLANNER_PROMPT
  parsing: str = "plan"
  model_name: str = "gpt-4"
  streaming: bool = False
  temperature: float = 0.7

class OptimizationPlanLLMConfig(AbstractLLMConfig):
  def __init__(self, task:str=None) -> None:
    self.mock_task = None # TODO wtf this is shit
    self.prompt: str = TP_PROMPTS[task] # TODO: this is bad. Only works for Optimization now
  avatar: str = "TP"
  parsing: str = "plan"
  model_name: str = "gpt-4"
  streaming: bool = False
  temperature: float = 0.7

class VLMConfig(AbstractLLMConfig):
  def __init__(self, task:str=None) -> None:
    self.mock_task = None # TODO wtf this is shit
    self.prompt: str = TP_PROMPTS[task] # TODO: this is bad. Only works for Optimization now
  model_name: str = "gpt-4-vision-preview"
  streaming: bool = False
  temperature: float = 0.6
  max_tokens: int = 150

class ObjectiveLLMConfig(AbstractLLMConfig):
  prompt: str = OBJECTIVE_DESIGNER_PROMPT
  parsing: str = "objective"
  model_name: str = "gpt-3.5-turbo"
  streaming: bool = False
  temperature: float = 0.7

class OptimizationLLMConfig(AbstractLLMConfig):
  prompt: str = OPTIMIZATION_DESIGNER_PROMPT
  parsing: str = "optimization"
  model_name: str = "gpt-3.5-turbo"
  streaming: bool = False
  temperature: float = 0.7

class NMPCObjectiveLLMConfig(AbstractLLMConfig):
  prompt: str = NMPC_OBJECTIVE_DESIGNER_PROMPT
  parsing: str = "objective"
  model_name: str = "gpt-4"
  streaming: bool = False
  temperature: float = 0.7

class NMPCOptimizationLLMConfig(AbstractLLMConfig):
  def __init__(self, task:str=None) -> None:
    self.mock_task = None#"OD_move_table"
    self.prompt: str = OD_PROMPTS[task] # TODO: this is bad. Only works for NMPC now
  avatar: str = "OD"
  parsing: str = "optimization"
  model_name: str = "gpt-4"
  streaming: bool = False
  temperature: float = 0.6



class BaseControllerConfig(AbstractControllerConfig):
  nx: int = 3
  nu: int = 3 
  T: int = 15
  dt: float = 0.1
  lu: float = -0.5 # lower bound on u
  hu: float = 0.5  # higher bound on u

class BaseNMPCConfig(AbstractControllerConfig):
  nx: int = 3
  nu: int = 3 
  T: int = 15
  dt: float = 0.05
  lu: float = -0.2 # lower bound on u
  hu: float = 0.2  # higher bound on u
  model_type: str = "discrete"
  penalty_term_cons: float = 1e7
  

class BaseRobotConfig(AbstractRobotConfig):
  def __init__(self, task:str=None) -> None:
    self.task: str = task
  name: str = "objective"
  tp_type: str = "plan_optimization"          # Task planner: ["plan_objective, plan_optimization"]
  od_type: str = "nmpc_optimization"          # Optimization Designer:  ["objective", "optimization"]
  controller_type: str = "optimization"  # Controller type:        ["objective", "optimization"]
  open_gripper_time: int = 15
  wait_s: float = 30. # wait time after a new MPC formualtion is applied



BaseLLMConfigs = {
  "plan_objective": ObjectivePlanLLMConfig,
  "plan_optimization": OptimizationPlanLLMConfig,
  "objective": ObjectiveLLMConfig,
  "optimization": OptimizationLLMConfig,
  "nmpc_objective": NMPCObjectiveLLMConfig,
  "nmpc_optimization": NMPCOptimizationLLMConfig
}

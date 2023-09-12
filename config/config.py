from core import AbstractControllerConfig, AbstractLLMConfig, AbstractRobotConfig, AbstractSimulaitonConfig
from prompts.stack import *


class ObjectivePlanLLMConfig(AbstractLLMConfig):
  prompt: str = OBJECTIVE_TASK_PLANNER_PROMPT
  parsing: str = "plan"
  model_name: str = "gpt-4"
  temperature: float = 0.7

class OptimizationPlanLLMConfig(AbstractLLMConfig):
  prompt: str = OPTIMIZATION_TASK_PLANNER_PROMPT
  parsing: str = "plan"
  model_name: str = "gpt-4"
  temperature: float = 0.7

class ObjectiveLLMConfig(AbstractLLMConfig):
  prompt: str = OBJECTIVE_DESIGNER_PROMPT
  parsing: str = "objective"
  model_name: str = "gpt-3.5-turbo"
  temperature: float = 0.7

class OptimizationLLMConfig(AbstractLLMConfig):
  prompt: str = OPTIMIZATION_DESIGNER_PROMPT
  parsing: str = "optimization"
  model_name: str = "gpt-3.5-turbo"
  temperature: float = 0.7

class NMPCObjectiveLLMConfig(AbstractLLMConfig):
  prompt: str = NMPC_OBJECTIVE_DESIGNER_PROMPT
  parsing: str = "objective"
  model_name: str = "gpt-4"
  temperature: float = 0.7

class NMPCOptimizationLLMConfig(AbstractLLMConfig):
  prompt: str = NMPC_OPTIMIZATION_DESIGNER_PROMPT
  parsing: str = "optimization"
  model_name: str = "gpt-4"
  temperature: float = 0.7

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
  name: str = "objective"
  tp_type: str = "plan_optimization"                   # Task planner: ["plan_objective, plan_optimization"]
  od_type: str = "nmpc_optimization"          # Optimization Designer:  ["objective", "optimization"]
  controller_type: str = "nmpc_optimization"  # Controller type:        ["objective", "optimization"]
  open_gripper_time: int = 15


class SimulationConfig(AbstractSimulaitonConfig):
  render: bool = True
  env_name: str = "CleanPlate"     # [Cubes, CleanPlate, Sponge, MoveTable]
  mock_plan: str = "clean_plate"  # [None, "stack", "pyramid", "L", "reverse", "clean_plate", "sponge", "move_table"]
  save_video: bool = True
  fps: int = 30 # only used if save_video = True
  dt: float = 0.05 # simulation timestep. Must be equal to that of controller

BaseLLMConfigs = {
  "plan_objective": ObjectivePlanLLMConfig,
  "plan_optimization": OptimizationPlanLLMConfig,
  "objective": ObjectiveLLMConfig,
  "optimization": OptimizationLLMConfig,
  "nmpc_objective": NMPCObjectiveLLMConfig,
  "nmpc_optimization": NMPCOptimizationLLMConfig
}

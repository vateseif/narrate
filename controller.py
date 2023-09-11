import torch
import do_mpc
import cvxpy as cp
import numpy as np
import casadi as ca
from itertools import chain
from typing import Dict, List, Tuple, Optional


from core import AbstractController
from llm import Objective, Optimization
from config.config import BaseControllerConfig, BaseNMPCConfig


class BaseController(AbstractController):

  def __init__(self, cfg=BaseControllerConfig()) -> None:
    super().__init__(cfg)
    # init linear dynamics
    self.init_dynamics()
    # init CVXPY problem
    self.init_problem()

  def init_dynamics(self):
    # dynamics
    self.A = np.zeros((self.cfg.nx, self.cfg.nx))
    self.B = np.eye(self.cfg.nx)
    self.Ad = np.eye(self.cfg.nx) + self.A * self.cfg.dt
    self.Bd = self.B * self.cfg.dt

  def init_problem(self):
    # variables
    self.x = cp.Variable((self.cfg.T+1, self.cfg.nx), name='x')
    self.u = cp.Variable((self.cfg.T, self.cfg.nu), name='u')
    # parameters
    self.x0 = cp.Parameter(self.cfg.nx, name="x0")
    self.xd = cp.Parameter(self.cfg.nx, name="xd")
    self.x0.value = np.zeros((self.cfg.nx,))
    self.xd.value = np.zeros((self.cfg.nx,))
    # cost
    self.xd_cost_T = sum([cp.norm(xt - self.xd) for xt in self.x])
    self.obj = cp.Minimize(self.xd_cost_T)
    # constraints
    self.cvx_constraints = self.init_cvx_constraints()
    # put toghether nominal MPC problem
    self.prob = cp.Problem(self.obj, self.cvx_constraints)

  def init_cvx_constraints(self):
    constraints = []
    # upper and lower bounds
    constraints += [self.u <= self.cfg.hu*np.ones((self.cfg.T, self.cfg.nu))]
    constraints += [self.u >= self.cfg.lu*np.ones((self.cfg.T, self.cfg.nu))]
    # initial cond
    constraints += [self.x[0] == self.x0]
    # dynamics
    for t in range(self.cfg.T):
      constraints += [self.x[t+1] == self.Ad @ self.x[t] + self.Bd @ self.u[t]]
    # bouns on state (gripper always above table)
    for t in range(self.cfg.T+1):
      constraints += [self.x[t][2] >= 0]
    return constraints

  def set_x0(self, x0: np.ndarray):
    self.x0.value = x0
    return

  def reset(self, x0: np.ndarray) -> None:
    self.init_problem()
    self.set_x0(x0)
    self.xd.value = x0
    return

  def _eval(self, code_str: str, x_cubes: Tuple[np.ndarray]):
    #TODO this is hard coded for when there are 4 cubes
    cube_1, cube_2, cube_3, cube_4 = x_cubes
    evaluated_code = eval(code_str, {
      "cp": cp,
      "np": np,
      "self": self,
      "cube_1": cube_1,
      "cube_2": cube_2,
      "cube_3": cube_3,
      "cube_4": cube_4
    })
    return evaluated_code

  def _solve(self):
    # solve for either uncostrained problem or for initial guess
    self.prob.solve(solver='MOSEK')
    return self.u.value[0]
  
  def step(self):
    return self._solve()



class ParametrizedRewardController(BaseController):

  def apply_gpt_message(self, gpt_message:str, x_cubes: Tuple[np.ndarray]):
    cube_1, cube_2, cube_3, cube_4 = x_cubes
    self.xd.value = eval(gpt_message)


class ObjectiveController(BaseController):

  def apply_gpt_message(self, objective: Objective, x_cubes: Tuple[np.ndarray]) -> None:
    # apply objective function
    obj = self._eval(objective.objective, x_cubes)
    self.obj = cp.Minimize(obj)
    # create new MPC problem
    self.prob = cp.Problem(self.obj, self.cvx_constraints)
    return 

class OptimizationController(BaseController):

  def apply_gpt_message(self, optimization: Optimization, x_cubes: Tuple[np.ndarray]) -> None:    
    # apply objective function
    obj = self._eval(optimization.objective, x_cubes)
    self.obj = cp.Minimize(obj)
    # apply constraints
    constraints = self.cvx_constraints
    for constraint in optimization.constraints:
      constraints += self._eval(constraint, x_cubes)
    # create new MPC problem
    self.prob = cp.Problem(self.obj, self.cvx_constraints)
    return 

class BaseNMPC(AbstractController):
  def __init__(self, cfg=BaseNMPCConfig()) -> None:
    super().__init__(cfg)

    # init linear dynamics
    self.init_dynamics()
    # init problem (cost and constraints)
    #self.init_problem()
    # init do_mpc problem
    self.init_controller()

    # init variables for python evaluation
    self.eval_variables = {"ca":ca, "cp":cp, "np":np} # python packages

  def init_dynamics(self):
    # inti do_mpc model
    self.model = do_mpc.model.Model(self.cfg.model_type) # TODO: add model_type to cfg
    # position (x, y, z)
    self.x = self.model.set_variable(var_type='_x', var_name='x', shape=(self.cfg.nx,1))
    # velocity (dx, dy, dz)
    self.dx = self.model.set_variable(var_type='_x', var_name='dx', shape=(self.cfg.nx,1))
    # controls (u1, u2, u3)
    self.u = self.model.set_variable(var_type='_u', var_name='u', shape=(self.cfg.nu,1))
    # system dynamics
    self.model.set_rhs('x', self.x + self.dx * self.cfg.dt)
    self.model.set_rhs('dx', self.u)
    # setup model
    self.model.setup()
    return

  def set_objective(self, mterm: ca.SX = ca.DM([[0]])): # TODO: not sure if ca.SX is the right one
    # objective terms
    mterm = mterm
    lterm = 0.4*mterm #ca.DM([[0]]) #
    # state objective
    self.mpc.set_objective(mterm=mterm, lterm=lterm)
    # input objective
    self.mpc.set_rterm(u=2)

    return

  def set_constraints(self, nlp_constraints: Optional[List[ca.SX]] = None):

    # base constraints (state)
    self.mpc.bounds['lower','_x', 'x'] = np.array([-100, -100, 0.0]) # stay above table
    # base constraints (input)
    self.mpc.bounds['upper','_u', 'u'] = self.cfg.hu * np.ones((self.cfg.nu, 1))  # input upper bound
    self.mpc.bounds['lower','_u', 'u'] = self.cfg.lu * np.ones((self.cfg.nu, 1))  # input lower bound

    if nlp_constraints == None: 
      return

    for i, constraint in enumerate(nlp_constraints):
      self.mpc.set_nl_cons(f'const{i}', expr=constraint, ub=0., 
                          soft_constraint=True, 
                          penalty_term_cons=self.cfg.penalty_term_cons)

    return

  def init_mpc(self):
    # init mpc model
    self.mpc = do_mpc.controller.MPC(self.model)
    # setup params
    setup_mpc = {'n_horizon': self.cfg.T, 't_step': self.cfg.dt, 'store_full_solution': False}
    # setup mpc
    self.mpc.set_param(**setup_mpc)
    self.mpc.settings.supress_ipopt_output() # => verbose = False
    return

  def init_controller(self):
    # init
    self.init_mpc()
    # set functions
    self.set_objective()
    self.set_constraints()
    # setup
    self.mpc.setup()
    self.mpc.set_initial_guess()
    

  def set_x0(self, x0: np.ndarray):
    self.mpc.x0 = np.concatenate((x0, np.zeros(3))) # concatenate velocity      

  def reset(self, x0: np.ndarray) -> None:
    # TODO
    #self.init_problem()
    self.set_x0(x0)
    #self.xd.value = x0
    return  


  def _eval(self, code_str: str, observation: Dict[str, np.ndarray], offset=0):
    #TODO the offset is still harcoded
    # put together variables for python code evaluation:
    # python packages | robot state (gripper) | environment observations
    eval_variables = self.eval_variables | {"x": self.x + offset} | observation
    # evaluate code
    evaluated_code = eval(code_str, eval_variables)
    return evaluated_code

  def _solve(self):
    # solve mpc at state x0
    u0 = self.mpc.make_step(self.mpc.x0)
    return u0.squeeze()

  def step(self):
    if not self.mpc.flags['setup']:
      return np.zeros(self.cfg.nu)
    return self._solve()

  
class ObjectiveNMPC(BaseNMPC):

  def apply_gpt_message(self, objective: Objective, observation: Dict[str, np.ndarray]) -> None:
    # init mpc newly
    self.init_mpc()
    # apply constraint function
    self.set_objective(self._eval(objective.objective, observation))
    # set base constraint functions
    self.set_constraints()
    # setup
    self.mpc.setup()
    self.mpc.set_initial_guess()
    return 

class OptimizationNMPC(BaseNMPC):

  def apply_gpt_message(self, optimization: Optimization, observation: Dict[str, np.ndarray]) -> None:
    # init mpc newly
    self.init_mpc()
    # apply constraint function
    self.set_objective(self._eval(optimization.objective, observation))
    # set base constraint functions
    gripper_offsets = [np.array([0., -0.048, 0.]), np.array([0., 0.048, 0.]), np.array([0., 0., 0.048])]
    constraints = [[*map(lambda x: self._eval(c, observation, x), gripper_offsets)] for c in optimization.constraints]
    self.set_constraints(list(chain(*constraints)))
    # setup
    self.mpc.setup()
    self.mpc.set_initial_guess()
    return 

ControllerOptions = {
  "parametrized": ParametrizedRewardController,
  "objective": ObjectiveController,
  "optimization": OptimizationController,
  "nmpc_objective": ObjectiveNMPC,
  "nmpc_optimization": OptimizationNMPC
}
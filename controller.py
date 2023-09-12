import do_mpc
import numpy as np
import casadi as ca
from time import sleep
from itertools import chain
from typing import Dict, List, Optional


from core import AbstractController
from llm import Objective, Optimization
from config.config import BaseNMPCConfig



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
    self.eval_variables = {"ca":ca, "np":np, "t":self.t} # python packages

    # gripper fingers offset for constraints 
    self.gripper_offsets = [np.array([0., -0.048, 0.]), np.array([0., 0.048, 0.]), np.array([0., 0., 0.048])]

  def init_dynamics(self):
    # inti do_mpc model
    self.model = do_mpc.model.Model(self.cfg.model_type) # TODO: add model_type to cfg
    # simulation time
    self.t = self.model.set_variable('parameter', 't')
    # position (x, y, z)
    self.x = self.model.set_variable(var_type='_x', var_name='x', shape=(self.cfg.nx,1))
    # pose (z axis)
    self.psi = self.model.set_variable(var_type='_x', var_name='psi', shape=(1,1))
    # velocity (dx, dy, dz)
    self.dx = self.model.set_variable(var_type='_x', var_name='dx', shape=(self.cfg.nx,1))
    # pose velocity
    self.dpsi = self.model.set_variable(var_type='_x', var_name='dpsi', shape=(1,1))
    # controls (u1, u2, u3)
    self.u = self.model.set_variable(var_type='_u', var_name='u', shape=(self.cfg.nu,1))
    self.u_psi = self.model.set_variable(var_type='_u', var_name='u_psi', shape=(1,1))
    # system dynamics
    self.model.set_rhs('x', self.x + self.dx * self.cfg.dt)
    self.model.set_rhs('psi', self.psi + self.dpsi * self.cfg.dt)
    self.model.set_rhs('dx', self.u)
    self.model.set_rhs('dpsi', self.u_psi)
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
    self.mpc.set_rterm(u=1, u_psi=1e-4)
    #self.mpc.set_rterm(u=1)
    return

  def set_constraints(self, nlp_constraints: Optional[List[ca.SX]] = None):

    # base constraints (state)
    self.mpc.bounds['lower','_x', 'x'] = np.array([-100, -100, 0.0]) # stay above table
    self.mpc.bounds['upper','_x', 'psi'] = np.pi/2 * np.ones((1, 1))   # rotation upper bound
    self.mpc.bounds['lower','_x', 'psi'] = -np.pi/2 * np.ones((1, 1))  # rotation lower bound
    # base constraints (input)
    self.mpc.bounds['upper','_u', 'u'] = self.cfg.hu * np.ones((self.cfg.nu, 1))  # input upper bound
    self.mpc.bounds['lower','_u', 'u'] = self.cfg.lu * np.ones((self.cfg.nu, 1))  # input lower bound
    self.mpc.bounds['upper','_u', 'u_psi'] = np.pi * np.ones((1, 1))   # input upper bound
    self.mpc.bounds['lower','_u', 'u_psi'] = -np.pi * np.ones((1, 1))  # input lower bound

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
    self.mpc.set_uncertainty_values(t=np.array([0.])) # init time to 0
    self.mpc.setup()
    self.mpc.set_initial_guess()

  def set_t(self, t:float):
    """ Update the simulation time of the MPC controller"""
    self.mpc.set_uncertainty_values(t=np.array([t]))

  def set_x0(self, x0: np.ndarray):
    self.mpc.x0 = np.concatenate((x0, np.zeros(1))) # TODO: 0 is dpsi hard-coded   

  def reset(self, x0: np.ndarray) -> None:
    # TODO
    #self.init_problem()
    self.set_x0(x0)
    #self.xd.value = x0
    return  


  def _eval(self, code_str: str, observation: Dict[str, np.ndarray], offset=np.zeros(3)):
    #TODO the offset is still harcoded
    # rotation matrix around z axis
    R = np.array([
      [ca.cos(self.psi), -ca.sin(self.psi), 0],
      [ca.sin(self.psi), ca.cos(self.psi), 0],
      [0, 0, 1]
    ])
    # put together variables for python code evaluation:
    # python packages | robot state (gripper) | environment observations
    eval_variables = self.eval_variables | {"x": self.x + R@offset, "dx": self.dx} | observation
    # evaluate code
    evaluated_code = eval(code_str, eval_variables)
    return evaluated_code

  def _solve(self):
    # solve mpc at state x0
    u0 = self.mpc.make_step(self.mpc.x0).squeeze()
    ee_displacement = u0[:3]
    psi_rotation = u0[-1]
    return np.concatenate((ee_displacement, np.array([0., 0., psi_rotation])))

  def step(self):
    if not self.mpc.flags['setup']:
      #sleep(1)
      return np.array([0., 0., 0., 0., 0., 0.]) # TODO change
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
    # NOTE use 1e-6 when doing task L 
    regulatization = 1 * ca.norm_2(self.dpsi)**2 #+ 0.1 * ca.norm_2(self.psi - np.pi/2)**2
    self.set_objective(self._eval(optimization.objective, observation) + regulatization)
    # set base constraint functions
    constraints = [[*map(lambda x: self._eval(c, observation, x), self.gripper_offsets)] for c in optimization.constraints]
    self.set_constraints(list(chain(*constraints)))
    # setup
    self.mpc.set_uncertainty_values(t=np.array([0.])) # TODO this is badly harcoded
    self.mpc.setup()
    self.mpc.set_initial_guess()
    return 

ControllerOptions = {
  "nmpc_objective": ObjectiveNMPC,
  "nmpc_optimization": OptimizationNMPC
}
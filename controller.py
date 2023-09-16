import do_mpc
import numpy as np
import casadi as ca
from time import sleep
from itertools import chain
from typing import Dict, List, Optional


from core import AbstractController
from llm import Objective, Optimization
from config.config import BaseNMPCConfig



class BaseController(AbstractController):
  def __init__(self, cfg=BaseNMPCConfig()) -> None:
    super().__init__(cfg)

    # init linear dynamics
    self.init_dynamics()
    # init do_mpc problem
    self.init_controller()

    # init gripper pose to keep track of measurements'
    self.gripper_pose = np.zeros(6) # [x, y, z, theta, gamma, psi]

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

  def set_x0(self, observation: Dict[str, np.ndarray]):
    obs = observation['robot_0']
    # store measurement of gripper pos
    self.gripper_pose = obs[:6]
    # init MPC x0 
    gripper_x = obs[:3]
    gripper_psi = np.array([obs[5]])
    gripper_dx = obs[6:9]
    self.mpc.x0 = np.concatenate((gripper_x, gripper_psi, gripper_dx, np.zeros(1))) # TODO: 0 is dpsi hard-coded   

  def reset(self, observation: Dict[str, np.ndarray]) -> None:
    """
      observation: robot observation from simulation containing position, angle and velocities 
    """
    # TODO
    self.set_x0(observation)
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
    psi_rotation = np.array([u0[-1]])
    #theta_gamma_rotation = (np.array([np.pi, 0.]) - self.gripper_pose[3:5]) # TODO: 2 is harcoded. Also init the desired theta_gamma
    theta_gamma_rotation = np.zeros(2)
    #print(self.gripper_pose[3:5])
    return [np.concatenate((ee_displacement, theta_gamma_rotation, psi_rotation))]

  def step(self):
    if not self.mpc.flags['setup']:
      return [np.zeros(6)] # TODO change
    return self._solve()


class BaseDualController(AbstractController):
  def __init__(self, cfg=BaseNMPCConfig()) -> None:
    super().__init__(cfg)
    # init linear dynamics
    self.init_dynamics()
    # init do_mpc problem
    self.init_controller()

    # init gripper pose to keep track of measurements'
    # [x, y, z, theta, gamma, psi]
    self.gripper_pose_left = np.zeros(6)
    self.gripper_pose_right = np.zeros(6)

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
    self.x_left = self.model.set_variable(var_type='_x', var_name='x_left', shape=(self.cfg.nx,1))
    # pose (z axis)
    self.psi_left = self.model.set_variable(var_type='_x', var_name='psi_left', shape=(1,1))
    # velocity (dx, dy, dz)
    self.dx_left = self.model.set_variable(var_type='_x', var_name='dx_left', shape=(self.cfg.nx,1))
    # pose velocity
    self.dpsi_left = self.model.set_variable(var_type='_x', var_name='dpsi_left', shape=(1,1))
    # controls (u1, u2, u3)
    self.u_left = self.model.set_variable(var_type='_u', var_name='u_left', shape=(self.cfg.nu,1))
    self.u_psi_left = self.model.set_variable(var_type='_u', var_name='u_psi_left', shape=(1,1))

    # position (x, y, z)
    self.x_right = self.model.set_variable(var_type='_x', var_name='x_right', shape=(self.cfg.nx,1))
    # pose (z axis)
    self.psi_right = self.model.set_variable(var_type='_x', var_name='psi_right', shape=(1,1))
    # velocity (dx, dy, dz)
    self.dx_right = self.model.set_variable(var_type='_x', var_name='dx_right', shape=(self.cfg.nx,1))
    # pose velocity
    self.dpsi_right = self.model.set_variable(var_type='_x', var_name='dpsi_right', shape=(1,1))
    # controls (u1, u2, u3)
    self.u_right = self.model.set_variable(var_type='_u', var_name='u_right', shape=(self.cfg.nu,1))
    self.u_psi_right = self.model.set_variable(var_type='_u', var_name='u_psi_right', shape=(1,1))
    # system dynamics
    self.model.set_rhs('x_left', self.x_left + self.dx_left * self.cfg.dt)
    self.model.set_rhs('psi_left', self.psi_left + self.dpsi_left * self.cfg.dt)
    self.model.set_rhs('dx_left', self.u_left)
    self.model.set_rhs('dpsi_left', self.u_psi_left)

    self.model.set_rhs('x_right', self.x_right + self.dx_right * self.cfg.dt)
    self.model.set_rhs('psi_right', self.psi_right + self.dpsi_right * self.cfg.dt)
    self.model.set_rhs('dx_right', self.u_right)
    self.model.set_rhs('dpsi_right', self.u_psi_right)
    # setup model
    self.model.setup()
    return

  def set_constraints(self, nlp_constraints: Optional[List[ca.SX]] = None):

    # base constraints (state)
    self.mpc.bounds['lower','_x', 'x_left'] = np.array([-3., -3., 0.0]) # stay above table
    #self.mpc.bounds['upper','_x', 'psi_left'] = np.pi/2 * np.ones((1, 1))   # rotation upper bound
    #self.mpc.bounds['lower','_x', 'psi_left'] = -np.pi/2 * np.ones((1, 1))  # rotation lower bound

    self.mpc.bounds['lower','_x', 'x_right'] = np.array([-3, -3, 0.0]) # stay above table
    #self.mpc.bounds['upper','_x', 'psi_right'] = np.pi/2 * np.ones((1, 1))   # rotation upper bound
    #self.mpc.bounds['lower','_x', 'psi_right'] = -np.pi/2 * np.ones((1, 1))  # rotation lower bound
    # base constraints (input)
    self.mpc.bounds['upper','_u', 'u_left'] = self.cfg.hu * np.ones((self.cfg.nu, 1))  # input upper bound
    self.mpc.bounds['lower','_u', 'u_left'] = self.cfg.lu * np.ones((self.cfg.nu, 1))  # input lower bound
    self.mpc.bounds['upper','_u', 'u_psi_left'] = np.pi * np.ones((1, 1))   # input upper bound
    self.mpc.bounds['lower','_u', 'u_psi_left'] = -np.pi * np.ones((1, 1))  # input lower bound

    self.mpc.bounds['upper','_u', 'u_right'] = self.cfg.hu * np.ones((self.cfg.nu, 1))  # input upper bound
    self.mpc.bounds['lower','_u', 'u_right'] = self.cfg.lu * np.ones((self.cfg.nu, 1))  # input lower bound
    self.mpc.bounds['upper','_u', 'u_psi_right'] = np.pi * np.ones((1, 1))   # input upper bound
    self.mpc.bounds['lower','_u', 'u_psi_right'] = -np.pi * np.ones((1, 1))  # input lower bound

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

  def set_t(self, t:float):
    """ Update the simulation time of the MPC controller"""
    self.mpc.set_uncertainty_values(t=np.array([t]))

  def init_controller(self):
    # init
    self.init_mpc()
    # set functions
    self.set_objective(ca.norm_2(self.psi_left)**2 + ca.norm_2(np.pi - self.psi_right)**2)
    self.set_constraints()
    # setup
    self.mpc.set_uncertainty_values(t=np.array([0.])) # init time to 0
    self.mpc.setup()
    self.mpc.set_initial_guess()
    #print("SET INITIAL GUESS")

  def reset(self, observation: Dict[str, np.ndarray]) -> None:
    """
      observation: robot observation from simulation containing position, angle and velocities 
    """
    # TODO
    self.set_x0(observation)
    return  

  def set_x0(self, observation: Dict[str, np.ndarray]):
    #print(observation)
    obs_left = observation['robot_0'] # TODO chaneg name?
    obs_right = observation['robot_1']
    # store measurement of gripper pos
    self.gripper_pose_left = obs_left[:6]
    # init MPC x0 
    gripper_x_left = obs_left[:3]
    gripper_psi_left = np.array([obs_left[5]])
    gripper_dx_left = obs_left[6:9]
    x0_left = np.concatenate((gripper_x_left, gripper_psi_left, gripper_dx_left, np.zeros(1)))
    # store measurement of gripper pos
    self.gripper_pose_right = obs_right[:6]
    # init MPC x0 
    gripper_x_right = obs_right[:3]
    gripper_psi_right = np.array([obs_right[5]])
    gripper_dx_right = obs_right[6:9]
    x0_right = np.concatenate((gripper_x_right, gripper_psi_right, gripper_dx_right, np.zeros(1)))

    x0 = np.concatenate((x0_left, x0_right))
    self.mpc.x0 = x0  # TODO: 0 is dpsi hard-coded   

  def set_objective(self, mterm: ca.SX = ca.DM([[0]])): # TODO: not sure if ca.SX is the right one
    # objective terms
    mterm = mterm
    lterm = 0.4*mterm #ca.DM([[0]]) #
    # state objective
    self.mpc.set_objective(mterm=mterm, lterm=lterm)
    # input objective
    self.mpc.set_rterm(u_left=1., u_right=1., u_psi_left=1, u_psi_right=1)
    #self.mpc.set_rterm(u=1)

  def _eval(self, code_str: str, observation: Dict[str, np.ndarray], offset=np.zeros(3)):
    #TODO the offset is still harcoded
    # rotation matrix around z axis
    R_left = np.array([
      [ca.cos(self.psi_left), -ca.sin(self.psi_left), 0],
      [ca.sin(self.psi_left), ca.cos(self.psi_left), 0],
      [0, 0, 1]
    ])

    R_right = np.array([
      [ca.cos(self.psi_right), -ca.sin(self.psi_right), 0],
      [ca.sin(self.psi_right), ca.cos(self.psi_right), 0],
      [0, 0, 1]
    ])
    # put together variables for python code evaluation:
    # python packages | robot state (gripper) | environment observations
    robots_states = {"x_left": self.x_left + R_left@offset, "dx_left": self.dx_left, "x_right": self.x_right + R_right@offset, "dx_right": self.dx_right} 
    eval_variables = self.eval_variables | robots_states | observation
    # evaluate code
    evaluated_code = eval(code_str, eval_variables)
    return evaluated_code

  def _solve(self):
    # solve mpc at state x0
    u0 = self.mpc.make_step(self.mpc.x0).squeeze()
    #print(u0)
    ee_displacement_left = u0[:3]
    psi_rotation_left = np.array([u0[3]])
    theta_gamma_rotation_left = np.zeros(2)
    action_left = np.concatenate((ee_displacement_left, theta_gamma_rotation_left, psi_rotation_left))

    ee_displacement_rigth = u0[4:7]
    psi_rotation_right = np.array([u0[7]])
    theta_gamma_rotation_rigth = np.zeros(2)
    action_rigth = np.concatenate((ee_displacement_rigth, theta_gamma_rotation_rigth, psi_rotation_right))
    
    return [action_left, action_rigth]

  def step(self):
    if not self.mpc.flags['setup']:
      return [np.zeros(6), np.zeros(6)]  # TODO change
    return self._solve()
  
class ObjectiveNMPC(BaseController):

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

class OptimizationNMPC(BaseController):

  def apply_gpt_message(self, optimization: Optimization, observation: Dict[str, np.ndarray]) -> None:
    # init mpc newly
    self.init_mpc()
    # apply constraint function
    # NOTE use 1e-6 when doing task L 
    regulatization = 1 * ca.norm_2(self.dpsi)**2 #+ 0.1 * ca.norm_2(self.psi - np.pi/2)**2
    self.set_objective(self._eval(optimization.objective, observation) + regulatization)
    # set base constraint functions
    constraints = [[*map(lambda const: self._eval(c, observation, const), self.gripper_offsets)] for c in optimization.constraints]
    self.set_constraints(list(chain(*constraints)))
    # setup
    self.mpc.set_uncertainty_values(t=np.array([0.])) # TODO this is badly harcoded
    self.mpc.setup()
    self.mpc.set_initial_guess()
    return 

class DualControllerOptimization(BaseDualController):

  def apply_gpt_message(self, optimization: Optimization, observation: Dict[str, np.ndarray]) -> None:
    # init mpc newly
    self.init_mpc()
    # apply constraint function
    # NOTE use 1e-6 when doing task L 
    self.set_objective(self._eval(optimization.objective, observation))
    # set base constraint functions
    constraints = [[*map(lambda const: self._eval(c, observation, const), self.gripper_offsets)] for c in optimization.constraints]
    self.set_constraints(list(chain(*constraints)))
    # setup
    self.mpc.set_uncertainty_values(t=np.array([0.])) # TODO this is badly harcoded
    self.mpc.setup()
    self.mpc.set_initial_guess()
    return 


ControllerOptions = {
  "nmpc_objective": ObjectiveNMPC,
  "nmpc_optimization": OptimizationNMPC,
  "dual": DualControllerOptimization
}
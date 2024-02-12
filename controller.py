import do_mpc
import numpy as np
import casadi as ca
from time import sleep
from itertools import chain
from typing import Dict, List, Optional, Tuple


from core import AbstractController
from llm import Objective, Optimization
from config.config import BaseNMPCConfig

class BaseController(AbstractController):

  def __init__(self, env_info:Tuple[List], cfg=BaseNMPCConfig) -> None:
    super().__init__(cfg)

    # init info of robots and objects
    self.robots_info, self.objects_info = env_info

    # init model dynamics
    self.init_model()

    # init controller
    self.init_controller()

    # init variables and expressions
    self.init_expressions()

    # gripper fingers offset for constraints 
    self.gripper_offsets = [np.array([0., -0.05, 0.]), np.array([0., 0.05, 0.]), np.array([0., 0., 0.05])]


  def init_model(self):
    # inti do_mpc model
    self.model = do_mpc.model.Model(self.cfg.model_type) 

    """ Parameter """
    # simulation time
    self.t = self.model.set_variable('parameter', 't')
    # position of objects
    self.objects = {} 
    for o in self.objects_info:
      if not o['name'].endswith("_orientation"):
        self.objects[o['name']] = self.model.set_variable(var_type='_p', var_name=o['name'], shape=(3,1))
      else:
        name = o['name'].replace("_orientation", "_psi")
        self.objects[name] = self.model.set_variable(var_type='_p', var_name=name)

    # home position [x y, z]
    self.x0 = {}
    #for r in self.robots_info:
    #  self.x0[f'x0{r["name"]}'] = self.model.set_variable(var_type='_p', var_name=f'x0{r["name"]}', shape=(3,1))
    
    # gripper pose [x, y, z, theta, gamma, psi]       
    self.pose = []    
    
    """ State and input variables """
    self.x = []       # gripper position (x,y,z)
    self.psi = []     # gripper psi (rotation around z axis)
    self.dx = []      # gripper velocity (vx, vy, vz)
    self.dpsi = []    # gripper rotational speed
    self.u = []       # gripper control (=velocity)
    self.u_psi = []   # gripper rotation control (=rotational velocity)
    
    

    for i, r in enumerate(self.robots_info):
      # position (x, y, z)
      self.x.append(self.model.set_variable(var_type='_x', var_name=f'x{r["name"]}', shape=(self.cfg.nx,1)))
      self.psi.append(self.model.set_variable(var_type='_x', var_name=f'psi{r["name"]}', shape=(1,1)))
      self.dx.append(self.model.set_variable(var_type='_x', var_name=f'dx{r["name"]}', shape=(self.cfg.nx,1)))
      self.dpsi.append(self.model.set_variable(var_type='_x', var_name=f'dpsi{r["name"]}', shape=(1,1)))
      self.u.append(self.model.set_variable(var_type='_u', var_name=f'u{r["name"]}', shape=(self.cfg.nu,1)))
      self.u_psi.append(self.model.set_variable(var_type='_u', var_name=f'u_psi{r["name"]}', shape=(1,1)))
      # system dynamics
      self.model.set_rhs(f'x{r["name"]}', self.x[i] + self.dx[i] * self.cfg.dt)
      self.model.set_rhs(f'psi{r["name"]}', self.psi[i] + self.dpsi[i] * self.cfg.dt)
      self.model.set_rhs(f'dx{r["name"]}', self.u[i])
      self.model.set_rhs(f'dpsi{r["name"]}', self.u_psi[i])
    
    # setup model
    self.model.setup()
  
  def set_objective(self, mterm: ca.SX = ca.DM([[0]])): # TODO: not sure if ca.SX is the right one
    # objective terms
    regularization = 0
    for i, r in enumerate(self.robots_info):
      #regularization += ca.norm_2(self.x[i] - r['x0'])**2
      regularization += .4 * ca.norm_2(self.dx[i])**2
      regularization += .4 * ca.norm_2(self.dpsi[i])**2#.4*ca.norm_2(ca.cos(self.psi[i]) - np.cos(r['euler0'][-1]))**2 # TODO 0.1 is harcoded
    mterm = mterm + regularization # TODO: add psi reference like this -> 0.1*ca.norm_2(-1-ca.cos(self.psi_right))**2
    lterm = 0.4*mterm
    # state objective
    self.mpc.set_objective(mterm=mterm, lterm=lterm)
    # input objective
    u_kwargs = {f'u{r["name"]}':1. for r in self.robots_info} | {f'u_psi{r["name"]}':1. for r in self.robots_info} 
    self.mpc.set_rterm(**u_kwargs)

  def set_constraints(self, nlp_constraints: Optional[List[ca.SX]] = None):

    for r in self.robots_info:
      # base constraints (state)
      self.mpc.bounds['lower','_x', f'x{r["name"]}'] = np.array([-3., -3., 0.0]) # stay above table
      #self.mpc.bounds['upper','_x', f'psi{r["name"]}'] = np.pi/2 * np.ones((1, 1))   # rotation upper bound
      #self.mpc.bounds['lower','_x', f'psi{r["name"]}'] = -np.pi/2 * np.ones((1, 1))  # rotation lower bound

      # base constraints (input)
      self.mpc.bounds['upper','_u', f'u{r["name"]}'] = self.cfg.hu * np.ones((self.cfg.nu, 1))  # input upper bound
      self.mpc.bounds['lower','_u', f'u{r["name"]}'] = self.cfg.lu * np.ones((self.cfg.nu, 1))  # input lower bound
      self.mpc.bounds['upper','_u', f'u_psi{r["name"]}'] = np.pi * np.ones((1, 1))   # input upper bound
      self.mpc.bounds['lower','_u', f'u_psi{r["name"]}'] = -np.pi * np.ones((1, 1))  # input lower bound

    if nlp_constraints == None: 
      return

    for i, constraint in enumerate(nlp_constraints):
      self.mpc.set_nl_cons(f'const{i}', expr=constraint, ub=0., 
                          soft_constraint=True, 
                          penalty_term_cons=self.cfg.penalty_term_cons)

  def init_mpc(self):
    # init mpc model
    self.mpc = do_mpc.controller.MPC(self.model)
    # setup params
    setup_mpc = {'n_horizon': self.cfg.T, 't_step': self.cfg.dt, 'store_full_solution': False}
    # setup mpc
    self.mpc.set_param(**setup_mpc)
    self.mpc.settings.supress_ipopt_output() # => verbose = False

  def init_controller(self):
    # init
    self.init_mpc()
    # set functions
    # TODO: should the regularization be always applied?
    self.set_objective()
    self.set_constraints()
    # setup
    self.mpc.set_uncertainty_values(t=np.array([0.])) # init time to 0
    self.mpc.setup()
    self.mpc.set_initial_guess()

  def init_expressions(self):
    # init variables for python evaluation
    self.eval_variables = {"ca":ca, "np":np, "t":self.t} # python packages

    self.R = [] # rotation matrix for angle around z axis
    for i in range(len(self.robots_info)):
      # rotation matrix
      self.R.append(np.array([[ca.cos(self.psi[i]), -ca.sin(self.psi[i]), 0],
                              [ca.sin(self.psi[i]), ca.cos(self.psi[i]), 0],
                              [0, 0, 1.]]))
      
  def _quaternion_to_euler_angle_vectorized2(self, quaternion):
      x, y, z, w = quaternion
      ysqr = y * y

      t0 = +2.0 * (w * x + y * z)
      t1 = +1.0 - 2.0 * (x * x + ysqr)
      X = np.arctan2(t0, t1)

      t2 = +2.0 * (w * y - z * x)

      t2 = np.clip(t2, a_min=-1.0, a_max=1.0)
      Y = np.arcsin(t2)

      t3 = +2.0 * (w * z + x * y)
      t4 = +1.0 - 2.0 * (ysqr + z * z)
      Z = np.arctan2(t3, t4)

      return np.array([X, Y, Z])

  def set_t(self, t:float):
    """ Update the simulation time of the MPC controller"""
    self.mpc.set_uncertainty_values(t=np.array([t]))

  def set_x0(self, observation: Dict[str, np.ndarray]):
    x0 = []
    self.pose = []
    for r in self.robots_info: # TODO set names instead of robot_0 in panda
      obs = observation[f'robot{r["name"]}'] # observation of each robot
      x = obs[:3]
      psi = np.array([obs[5]])
      dx = obs[6:9]
      x0.append(np.concatenate((x, psi, dx, [0]))) # TODO dpsi is harcoded to 0 here
      self.pose.append(obs[:6])
    # set x0 in MPC
    self.mpc.x0 = np.concatenate(x0)

  def init_states(self, observation:Dict[str, np.ndarray], t:float):
    """ Set the values the MPC initial states and variables """
    # set mpc x0
    self.set_x0(observation)
    # set variable parameters
    parameters = {'t': [t]}
    parameters = parameters | {o['name']: [observation[o['name']]] for o in self.objects_info if not o["name"].endswith("_orientation")}
    parameters = parameters | {o['name'].replace("_orientation", "_psi"): [self._quaternion_to_euler_angle_vectorized2(observation[o['name']])[-1]] for o in self.objects_info if o["name"].endswith("_orientation")}
    #print(parameters)
    self.mpc.set_uncertainty_values(**parameters)

  def reset(self, observation: Dict[str, np.ndarray], t:float = 0) -> None:
    """
      observation: robot observation from simulation containing position, angle and velocities 
    """
    # TODO
    self.init_states(observation, t)
    return

  def _eval(self, code_str: str, observation: Dict[str, np.ndarray], offset=np.zeros(3)):
    #TODO the offset is still harcoded
    # put together variables for python code evaluation:    
    
    # initial state of robots before applying any action
    x0 = {f'x0{r["name"]}': observation[f'robot{r["name"]}'][:3] for r in self.robots_info} 
    # robot variable states (decision variables in the optimization problem)
    robots_states = {}
    for i, r in enumerate(self.robots_info):
      robots_states[f'x{r["name"]}'] = self.x[i] + self.R[i]@offset
      robots_states[f'dx{r["name"]}'] = self.dx[i]
      robots_states[f'psi{r["name"]}'] = self.psi[i]
    
    eval_variables = self.eval_variables | robots_states | self.objects | x0
    # evaluate code
    evaluated_code = eval(code_str, eval_variables)
    return evaluated_code

  def _solve(self) -> List[np.ndarray]:
    """ Returns a list of conntrols, 1 for each robot """
    # solve mpc at state x0
    u0 = self.mpc.make_step(self.mpc.x0).squeeze()
    # compute action for each robot
    action = []
    for i in range(len(self.robots_info)):
      ee_displacement = u0[4*i:4*i+3]     # positon control
      theta_regularized = self.pose[i][3] if self.pose[i][3]>=0 else self.pose[i][3] + 2*np.pi 
      theta_rotation = [(np.pi - theta_regularized)*1.5]
      gamma_rotation = [-self.pose[i][4] * 1.5]  # P control for angle around y axis # TODO: 1. is a hardcoded gain
      psi_rotation = [u0[4*i+3]]            # rotation control
      action.append(np.concatenate((ee_displacement, theta_rotation, gamma_rotation, psi_rotation)))
    
    return action

  def step(self):
    if not self.mpc.flags['setup']:
      return [np.zeros(6) for i in range(len(self.robots_info))]  # TODO 6 is hard-coded here
    return self._solve()


class ObjectiveController(BaseController):

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

class OptimizationController(BaseController):

  def apply_gpt_message(self, optimization: Optimization, observation: Dict[str, np.ndarray]) -> None:
    # init mpc newly
    self.init_mpc()
    # apply constraint function
    # NOTE use 1e-6 when doing task L 
    regulatization = 0#1 * ca.norm_2(self.dpsi)**2 #+ 0.1 * ca.norm_2(self.psi - np.pi/2)**2
    self.set_objective(self._eval(optimization.objective, observation) + regulatization)
    # set base constraint functions
    constraints = []
    # positive equality constraint
    constraints += [self._eval(c, observation) for c in optimization.equality_constraints]
    # negative equality constraint
    constraints += [-self._eval(c, observation) for c in optimization.equality_constraints]
    # inequality constraints
    inequality_constraints = [[*map(lambda const: self._eval(c, observation, const), self.gripper_offsets)] for c in optimization.inequality_constraints]
    constraints += list(chain(*inequality_constraints))
    # set constraints
    self.set_constraints(constraints)
    # setup
    self.mpc.set_uncertainty_values(t=np.array([0.])) # TODO this is badly harcoded
    self.mpc.setup()
    self.mpc.set_initial_guess()
    return 

ControllerOptions = {
  "objective": ObjectiveController,
  "optimization": OptimizationController
}



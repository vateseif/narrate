import do_mpc
import numpy as np
import casadi as ca
from time import time
from itertools import chain
from typing import Dict, List, Optional, Tuple

from core import AbstractController
from config.config import ControllerConfig


class Object:
	def __init__(self, name:str, position:np.ndarray=np.zeros((3,1)), psi:float=0., size:float=0.) -> None:
		self.name = name
		self.position = position
		self.psi = psi
		self.size = size

class Controller(AbstractController):

	def __init__(self, env_info:Tuple[List], cfg=ControllerConfig) -> None:
		super().__init__(cfg)

		# init info of robots and objects
		self.robots_info, self.objects_info = env_info

		

		# gripper fingers offset for constraints 
		self.gripper_offsets = [(np.array([0., -0.048, 0.003]), 0.013), (np.array([0., 0.048, 0.003]), 0.013), 
								(np.array([0., 0.0, 0.06]), 0.025), (np.array([0., -0.045, 0.06]), 0.025),
								(np.array([0., 0.045, 0.06]), 0.025), (np.array([0., -0.09, 0.06]), 0.025), (np.array([0., 0.09, 0.06]), 0.025)
								]
		
		self.gripper_offsets_load = [(np.array([0., 0., 0.]), 0.03)]


		self.gripper_closed = False

		# init controller
		self.setup_controller()

	def init_model(self):
		# inti do_mpc model
		self.model = do_mpc.model.Model(self.cfg.model_type) 

		# simulation time
		self.t = self.model.set_variable('parameter', 't')
		# position of objects
		self.objects = {} 
		for o in self.objects_info:
			position = self.model.set_variable(var_type='_p', var_name=o['name']+'_position', shape=(3,1))
			psi = self.model.set_variable(var_type='_p', var_name=o['name']+'_psi')
			size = self.model.set_variable(var_type='_p', var_name=o['name']+'_size')
			obj = Object(o['name'], position, psi, size)
			self.objects[o['name']] = obj

		# gripper pose [x, y, z, theta, gamma, psi]       
		self.pose = []    
		
		self.x = []       # gripper position (x,y,z)
		self.psi = []     # gripper psi (rotation around z axis)
		self.dx = []      # gripper velocity (vx, vy, vz)
		self.dpsi = []    # gripper rotational speed
		self.u = []       # gripper control (=velocity)
		self.u_psi = []   # gripper rotation control (=rotational velocity)
		self.cost = 1.    # cost function
		self.prev_cost = float('inf') # previous cost function
		self.solve_time = 0. # time to solve the optimization problem
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

	def setup_controller(self, optimization={"objective":None, "equality_constraints":[], "inequality_constraints":[]}):
		self.init_model()
		# init cost function
		self.model.set_expression('cost', self._eval(optimization["objective"]))
		# setup model
		self.model.setup()
		# init variables and expressions
		self.init_expressions()
		# init
		self.init_mpc()
		# set functions
		# TODO: should the regularization be always applied?
		regularization = 0#1 * ca.norm_2(self.dpsi)**2 #+ 0.1 * ca.norm_2(self.psi - np.pi/2)**2
		self.set_objective(self._eval(optimization["objective"]) + regularization)
		# set base constraint functions
		constraints = []
		# positive equality constraint
		constraints += [self._eval(c) for c in optimization["equality_constraints"]]
		# negative equality constraint
		constraints += [-self._eval(c) for c in optimization["equality_constraints"]]
		# inequality constraints
		gripper_offsets = self.get_gripper_offsets()
		inequality_constraints = [[*map(lambda const: self._eval(c, const), gripper_offsets)] for c in optimization["inequality_constraints"]]
		constraints += list(chain(*inequality_constraints))
		# set constraints
		self.set_constraints(constraints)
		# setup
		self.mpc.set_uncertainty_values(t=np.array([0.])) # init time to 0
		self.mpc.setup()
		self.mpc.set_initial_guess()

	def _normalize_angle(self, angle):
		"""
		Normalize an angle to be within the range [-pi, pi].
		
		Parameters:
		angle (float): The angle in radians to be normalized.
		
		Returns:
		float: The normalized angle within the range [-pi, pi].
		"""
		normalized_angle = np.arctan2(np.sin(angle), np.cos(angle))
		# Check if the angle is outside the range [-pi/2, pi/2] and adjust
		if normalized_angle > np.pi/2:
			normalized_angle -= np.pi
		elif normalized_angle < -np.pi/2:
			normalized_angle += np.pi
		return normalized_angle
	
	def set_objective(self, mterm: ca.SX=ca.DM([[0]])): # TODO: not sure if ca.SX is the right one
		# objective terms
		regularization = 0
		for i, r in enumerate(self.robots_info):
			#regularization += ca.norm_2(self.x[i] - (np.array([0,0,0.2])))**2
			regularization += .1 * ca.norm_2(self.dx[i])**2
			if self.cfg.task == "Cubes":
				regularization += .0002 * ca.norm_2(ca.sin(self.psi[i]) * ca.cos(self.psi[i]))**2
			else:
				regularization += .4 * ca.norm_2(ca.sin(self.psi[i]))**2
		mterm = mterm + regularization # TODO: add psi reference like this -> 0.1*ca.norm_2(-1-ca.cos(self.psi_right))**2
		lterm = 2*mterm
		# state objective
		self.mpc.set_objective(mterm=mterm, lterm=lterm)
		# input objective
		u_kwargs = {f'u{r["name"]}':0.5 for r in self.robots_info} | {f'u_psi{r["name"]}':1e-5 for r in self.robots_info} 
		self.mpc.set_rterm(**u_kwargs)

	def set_constraints(self, nlp_constraints: Optional[List[ca.SX]] = None):

		for r in self.robots_info:
			# base constraints (state)
			self.mpc.bounds['lower','_x', f'x{r["name"]}'] = np.array([-3., -3., 0.0]) # stay above table
			self.mpc.bounds['upper','_x', f'psi{r["name"]}'] = np.pi * 0.55 * np.ones((1, 1))   # rotation upper bound
			self.mpc.bounds['lower','_x', f'psi{r["name"]}'] = -np.pi * 0.55 * np.ones((1, 1))  # rotation lower bound

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


	def init_expressions(self):
		# init variables for python evaluation
		self.eval_variables = {"ca":ca, "np":np} # python packages

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

	def _set_x0(self, observation: Dict[str, np.ndarray]):
		x0 = []
		self.pose = []
		for r in self.robots_info: # TODO set names instead of robot_0 in panda
			obs = observation[f'robot{r["name"]}'] # observation of each robot
			x = obs[:3]
			psi = self._normalize_angle(np.array([obs[5]]))
			dx = obs[6:9]
			x0.append(np.concatenate((x, psi, dx, [0]))) # TODO dpsi is harcoded to 0 here
			self.pose.append(obs[:6])
		# set x0 in MPC
		self.mpc.x0 = np.concatenate(x0)

	def init_states(self, observation:Dict[str, np.ndarray], t:float, gripper_closed:bool=False):
		""" Set the values the MPC initial states and variables """
		self.gripper_closed = gripper_closed
		self.observation = observation
		# set mpc x0
		self._set_x0(observation)
		# set variable parameters
		parameters = {'t': [t]}
		parameters = parameters | {o['name']+'_position': [observation[o['name']]['position']] for o in self.objects_info}
		parameters = parameters | {o['name']+'_size': [observation[o['name']]['size']] for o in self.objects_info}
		parameters = parameters | {o['name']+'_psi': [self._quaternion_to_euler_angle_vectorized2(observation[o['name']]['orientation'])[-1]] for o in self.objects_info if o["name"].endswith("_orientation")}
		#print(parameters)
		self.mpc.set_uncertainty_values(**parameters)

	def get_gripper_offsets(self):
		if self.gripper_closed:
			gripper_offset = self.gripper_offsets + self.gripper_offsets_load	
		else:
			gripper_offset = self.gripper_offsets
		return gripper_offset

	def reset(self) -> None:
		"""
			observation: robot observation from simulation containing position, angle and velocities 
		"""
		# TODO
		self.setup_controller()
		return

	def _eval(self, code_str: str, offset=(np.zeros(3), 0.)):
		#TODO the offset is still harcoded
		# put together variables for python code evaluation:    
		if code_str == None: return ca.SX(0)
		
		# parse offset
		collision_xyz, collision_radius = offset
		# initial state of robots before applying any action
		x0 = {f'x0{r["name"]}': self.observation[f'robot{r["name"]}'][:3] for r in self.robots_info} 
		# robot variable states (decision variables in the optimization problem)
		robots_states = {}
		for i, r in enumerate(self.robots_info):
			robots_states[f'x{r["name"]}'] = self.x[i] + self.R[i]@collision_xyz
			robots_states[f'dx{r["name"]}'] = self.dx[i]
			robots_states[f'psi{r["name"]}'] = self.psi[i]
		
		eval_variables = self.eval_variables | robots_states | self.objects | x0 | {'t': self.t}
		# evaluate code
		evaluated_code = eval(code_str, eval_variables) + collision_radius
		return evaluated_code

	def _solve(self) -> List[np.ndarray]:
		""" Returns a list of conntrols, 1 for each robot """
		# solve mpc at state x0
		t0 = time()
		u0 = self.mpc.make_step(self.mpc.x0).squeeze()
		self.solve_time = time() - t0
		# compute action for each robot
		action = []
		for i in range(len(self.robots_info)):
			ee_displacement = u0[4*i:4*i+3]     # positon control
			theta_regularized = self.pose[i][3] if self.pose[i][3]>=0 else self.pose[i][3] + 2*np.pi 
			theta_rotation = [(np.pi - theta_regularized)*1.5]
			gamma_rotation = [-self.pose[i][4] * 1.5]  # P control for angle around y axis # TODO: 1. is a hardcoded gain
			psi_rotation = [u0[4*i+3]]            # rotation control
			action.append(np.concatenate((ee_displacement, theta_rotation, gamma_rotation, psi_rotation)))

		self.prev_cost = self.cost
		self.cost = self.mpc.data['_aux'][-1][-1]

		return action

	def step(self):
		if not self.mpc.flags['setup']:
			return [np.zeros(6) for i in range(len(self.robots_info))]  # TODO 6 is hard-coded here
		return self._solve()
	
	def retrieve_trajectory(self):
		trajectory = []
		try:
			for _x in self.mpc.opt_x_num['_x', :, 0, 0]:
				_x = _x.toarray().flatten()
				trajectory.append(_x[:3])
		except:
			pass
		return trajectory


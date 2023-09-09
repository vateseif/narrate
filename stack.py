import threading
import panda_gym
import numpy as np
import gymnasium as gym
#from gpt import GPTAgent
from controller import Controller

from time import sleep


class Sim:
  def __init__(self) -> None:
    # params
    self.n_episodes = 100
    self.max_episode_length = 100000

    # env
    self.env = gym.make("PandaStack-v3", render_mode="human", max_episode_steps=self.max_episode_length)
    self.observation, info = self.env.reset()


    self.functions = {
      'x_cube1': self._x_cube1,
      'x_cube2': self._x_cube2
    }

  def reset(self):
    # reset pand env
    observation, _ = self.env.reset()
    # store observation
    self.x0 = observation["observation"][0:3]
    self.x_cube = observation["achieved_goal"][0:3]
    self.x_cube_target = observation["desired_goal"][0:3]
    # reset controller
    robot.set_x0(self.x0)
    robot.set_xd(self.x0)
    robot.open_gripper()

  def step(self, action: np.ndarray):
    self.observation, _, terminated, truncated, _ = self.env.step(action)
    # store observation
    self.x0 = self.observation["observation"][0:3]

    self.x_cube1 = self.observation["achieved_goal"][0:3]
    self.x_cube2 = self.observation["achieved_goal"][3:6]
    #self.x_cube_target = observation["desired_goal"][0:3]
    return terminated or truncated

  def _x0(self):
    return self.x0

  def _x_cube1(self):
    return self.x_cube1

  def _x_cube2(self):
    return self.x_cube2

  def _x_cube_target(self):
    return self.x_cube_target
  
  def run(self):
    for _ in range(self.n_episodes):
      self.reset()
      while True:
        # update controller
        robot.set_x0(self.x0)
        # compute action
        action = robot.step()
        action = np.hstack((action, robot.gripper))
        # step env
        done = self.step(action)
        
        sleep(0.1)
        if done:
            break

    self.env.close()


# controller
robot = Controller()
# simulator
sim = Sim()  

# agent
#agent = GPTAgent(robot, sim)

thread = threading.Thread(target=sim.run)
thread.daemon = True  # Set the thread as a daemon (will exit when the main program ends)
thread.start()


starts_msg = """
Your goal is that of stacking 2 cubes that are on the table on top of each other. The position of the center of the first cube is `x_cube1` while the position of center of the second is `x_cube2`. The cubes have size 0.04m.
"""
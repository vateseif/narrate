import os
import threading
import numpy as np
from time import sleep
from typing import Optional
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter

from llm import Plan
from robot import BaseRobot
from core import AbstractSimulation, BASE_DIR
from config.config import SimulationConfig
from mocks.mocks import nmpcMockOptions # TODO 


class Simulation(AbstractSimulation):
  def __init__(self, cfg=SimulationConfig()) -> None:
    super().__init__(cfg)

    # TODO: account for multiple robots
    self.robot = BaseRobot()
    # count number of tasks solved from a plan 
    self.task_counter = 0
    # bool for stopping simulation
    self.stop_thread = False
    # init list of RGB frames if wanna save video
    if self.cfg.save_video:
      self.frames_list = []
      self.video_name = f"{self.cfg.env_name}_{self.cfg.mock_plan}_{datetime.now().strftime('%d-%m-%Y_%H:%M:%S')}"
      self.video_path = os.path.join(BASE_DIR, f"videos/{self.video_name}.mp4")

  def _reinit_robot(self):
    # x0 = [x, y, z, psi, dx, dy, dz]
    gripper_x = self.observation['robot_0'][:3]
    gripper_psi = np.array([self.observation['robot_0'][5]])
    gripper_dx = self.observation['robot_0'][6:9]
    x0 = np.concatenate((gripper_x, gripper_psi, gripper_dx))
    self.robot.reset(x0)

  def reset(self):
    # reset pand env
    self.observation = self.env.reset()
    # reset controller
    self._reinit_robot()
    # count number of tasks solved from a plan 
    self.task_counter = 0
    # init list of RGB frames if wanna save video
    if self.cfg.save_video:
      self.frames_list = []

  def create_plan(self, user_task:str, wait_s:Optional[int]=None): 
    self.plan = self.robot.create_plan(user_task) if self.cfg.mock_plan is None else nmpcMockOptions[self.cfg.mock_plan]
    print(f"\33[92m {self.plan.tasks} \033[0m \n")
    if wait_s is not None:
      for _ in self.plan.tasks:
        self.next_task()
        sleep(wait_s)

  def step(self):
    # update controller (i.e. set the current gripper position)
    self._reinit_robot()
    # compute action
    action = [self.robot.step()] # TODO: this is a list because the env may have multiple robots
    # apply action
    self.observation, _, done, _ = self.env.step(action)
    # store RGB frames if wanna save video
    if self.cfg.save_video:
      frame = self.env.render("rgb_array")
      self.frames_list.append(frame)

    return done

  def close(self):
    # close environment
    #self.thread.join()
    self.stop_thread = True
    self.thread.join()
    # init list of RGB frames if wanna save video
    if self.cfg.save_video:
      self._save_video()
      

  def _save_video(self):
    # Create a figure and axis for plotting.
    fig, ax = plt.subplots()
    # Initialize the writer for saving the video.
    writer = FFMpegWriter(fps=self.cfg.fps)
    # Iterate through your list of RGB images and add them to the video.
    with writer.saving(fig, self.video_path, dpi=200):
      for i, rgb_image in enumerate(self.frames_list):
        ax.clear()  # Clear the previous frame.
        ax.imshow(rgb_image)  # Display the current RGB image.
        plt.axis("off")  # Turn off axis labels and ticks.
        # You can optionally add a title or annotation to each frame.
        ax.set_title(f"Frame {i + 1}")
        writer.grab_frame()  # Save the current frame to the video.

  def _solve_task(self, plan:str):
    self.robot.next_plan(plan, self.observation)
    return

  def next_task(self):
    self._solve_task(self.plan.tasks[self.task_counter])
    self.task_counter += 1

  def _run(self):
    self.reset()
    while not self.stop_thread:
      # step env
      done = self.step()
      sleep(0.1)
      if done:
          break

    self.env.close()

  def run(self):
    """ Executes self._run() in a separate thread"""
    self.thread = threading.Thread(target=self._run)
    self.thread.daemon = True  # Set the thread as a daemon (will exit when the main program ends)
    self.thread.start()
    return
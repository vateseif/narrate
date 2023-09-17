import os
import sys
import threading
import numpy as np
from time import sleep
from typing import Optional
from datetime import datetime
from streamlit import spinner
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter

from robot import BaseRobot
from core import AbstractSimulation, BASE_DIR
from config.config import SimulationConfig, BaseRobotConfig


class Simulation(AbstractSimulation):
  def __init__(self, cfg=SimulationConfig()) -> None:
    super().__init__(cfg)

    # simulation time
    self.t = 0.
    self.robot = BaseRobot(self.env.robots_info,BaseRobotConfig(self.cfg.task))
    # count number of tasks solved from a plan 
    self.task_counter = 0
    # bool for stopping simulation
    self.stop_thread = False
    # init list of RGB frames if wanna save video
    if self.cfg.save_video:
      self.frames_list = []
      self.video_name = f"{self.cfg.env_name}_{self.cfg.task}_{datetime.now().strftime('%d-%m-%Y_%H:%M:%S')}"
      self.video_path = os.path.join(BASE_DIR, f"videos/{self.video_name}.mp4")

  def _reinit_robot(self):
    """ Update simulation time and current state of MPC controller"""
    self.robot.set_t(self.t)
    # set x0 to measurements
    self.robot.set_x0(self.observation)

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

  def create_plan(self, user_task:str, solve:bool=False): 
    sleep(1)
    self.task_counter = 0
    self.plan = self.robot.create_plan(user_task)
    if solve:
      self.execute_plan()

  def execute_plan(self):
    for _ in self.plan.tasks:
      self.next_task()

  def step(self):
    # increase timestep
    self.t += self.cfg.dt
    # update controller (i.e. set the current gripper position)
    self._reinit_robot()
    # compute action
    action = self.robot.step() # TODO: this is a list because the env may have multiple robots
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
    # exit
    sys.exit()
      

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
    wait_s = self.robot.next_plan(plan, self.observation)
    with spinner("Executing MPC code..."):
      sleep(wait_s)

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
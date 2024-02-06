import os
import sys
import cv2
import gym
import panda_gym
import threading
import numpy as np
from time import sleep
from typing import Optional
from datetime import datetime
from streamlit import spinner

from robot import BaseRobot
from core import AbstractSimulation, BASE_DIR
from config.config import SimulationConfig, BaseRobotConfig


class Simulation(AbstractSimulation):
  def __init__(self, cfg=SimulationConfig()) -> None:
    #super().__init__(cfg)

    self.cfg = cfg
    # init env
    self.env = gym.make(f"PandaCubes-v2", render=True)#gym.make(f"Panda{cfg.env_name}-v2", render=cfg.render)
    # init robots
    # count number of tasks solved from a plan 
    self.task_counter = 0

    # simulation time
    self.t = 0.
    env_info = (self.env.robots_info, self.env.objects_info)
    self.robot = BaseRobot(env_info,BaseRobotConfig(self.cfg.task))
    # count number of tasks solved from a plan 
    self.task_counter = 0
    # bool for stopping simulation
    self.stop_thread = False
    # whether to save frame (initialized to false)
    self.save_video = False
    # init list of RGB frames if wanna save video
    self.frames_list = []
    self.video_name = f"{self.cfg.env_name}_{self.cfg.task}_{datetime.now().strftime('%d-%m-%Y_%H:%M:%S')}"
    self.video_path = os.path.join(BASE_DIR, f"videos/{self.video_name}.mp4")

  def _reinit_robot(self):
    """ Update simulation time and current state of MPC controller"""
    self.robot.init_states(self.observation, self.t)

  def reset(self):
    # reset pand env
    self.observation = self.env.reset()
    # reset controller
    self._reinit_robot()
    # count number of tasks solved from a plan 
    self.task_counter = 0
    # init list of RGB frames if wanna save video
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
    if self.save_video:
      frame = self.env.render("rgb_array")
      self.frames_list.append(frame)

    return done

  def close(self):
    # close environment
    #self.thread.join()
    self.stop_thread = True
    self.thread.join()
    # init list of RGB frames if wanna save video
    if self.save_video:
      self._save_video()
    # exit
    sys.exit()
      

  def _save_video(self):
    # Define the parameters
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    width, height = 1920, 1920  # Adjust as needed
    # Create a VideoWriter object
    out = cv2.VideoWriter(self.video_path, fourcc, self.cfg.fps, (width, height))
    # Write frames to the video
    for frame in self.frames_list:
      # Ensure the frame is in the correct format (RGBA)
      if frame.shape[2] == 3:
          frame = cv2.cvtColor(frame, cv2.COLOR_RGB2RGBA)
      # Convert the frame to BGR format (required by VideoWriter)
      frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
      out.write(frame_bgr)
    # Release the VideoWriter
    out.release()

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
    #self.thread.daemon = True  # Set the thread as a daemon (will exit when the main program ends)
    self.thread.start()
    return
  

if __name__=="__main__":
  s = Simulation()
  s.run()
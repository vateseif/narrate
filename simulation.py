import os
import sys
import cv2
import gym
import panda_gym
import numpy as np
from time import sleep
from typing import Optional
from datetime import datetime
import asyncio
from aiohttp import web
from tqdm import tqdm

from robot import BaseRobot
from core import AbstractSimulation, BASE_DIR
from config.config import SimulationConfig, BaseRobotConfig


class Simulation(AbstractSimulation):
    def __init__(self, cfg=SimulationConfig()) -> None:
        #super().__init__(cfg)

        self.cfg = cfg
        # init env
        self.env = gym.make(f"Panda{cfg.env_name}-v2", render=cfg.render)
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

    def create_plan(self, user_task:str) -> str: 
        self.task_counter = 0
        self.plan = self.robot.create_plan(user_task)
        return self.plan.pretty_print()

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
            frame = np.array(self.env.render("rgb_array")).reshape(self.cfg.width, self.cfg.height, 4).astype(np.uint8)
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
        # Create a VideoWriter object
        out = cv2.VideoWriter(self.video_path, fourcc, self.cfg.fps, (self.cfg.width, self.cfg.height))
        # Write frames to the video
        for frame in tqdm(self.frames_list):
            # Ensure the frame is in the correct format (RGBA)
            if frame.shape[2] == 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2RGBA)
            # Convert the frame to BGR format (required by VideoWriter)
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
            out.write(frame_bgr)
        # Release the VideoWriter
        out.release()

    def _solve_task(self, plan:str):
        AI_response = self.robot.next_plan(plan, self.observation)
        return AI_response

    async def http_solve_task_handler(self, request):
        data = await request.json()
        user_task = data.get('task')
        AI_response = self._solve_task(user_task)
        return web.json_response({"avatar": "OD", "response": AI_response})

    async def http_plan_handler(self, request):
        data = await request.json()
        user_task = data.get('task')
        AI_response = self.create_plan(user_task)
        return web.json_response({"avatar": "TP", "response": AI_response})
    
    async def http_save_recording_handler(self, request):
        self.save_video = False
        self._save_video()
        return web.json_response({"response": "Recording saved"})
    
    async def http_start_recording_handler(self, request):      
        self.save_video = True
        return web.json_response({"response": "Recording started"})
    
    async def http_cancel_recording_handler(self, request):
        self.save_video = False
        self.frames_list = []
        return web.json_response({"response": "Recording cancelled"})
    
    async def http_execute_next_task_handler(self, request):
        next_task : str = self.plan.tasks[self.task_counter]
        self.task_counter += 1
        AI_response = self._solve_task(next_task)
        return web.json_response({"avatar": "OD", "response": AI_response})

    async def main(self, app):
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, 'localhost', 8080)
        await site.start()

        await self._run()

    async def _run(self):
        self.reset()
        while not self.stop_thread:
            done = self.step()
            await asyncio.sleep(0.05)
            if done:
                break
        self.env.close()

    def run(self):
        app = web.Application()
        app.add_routes([
            web.post('/create_plan', self.http_plan_handler),
            web.post('/solve_task', self.http_solve_task_handler),
            web.get('/save_recording', self.http_save_recording_handler),
            web.get('/start_recording', self.http_start_recording_handler),
            web.get('/cancel_recording', self.http_cancel_recording_handler),
            web.get('/execute_next_task', self.http_execute_next_task_handler)
        ])

        asyncio.run(self.main(app))
  

if __name__=="__main__":
    s = Simulation()
    s.run()
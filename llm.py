from time import sleep
from typing import List, Optional
from core import AbstractLLM, AbstractLLMConfig
from mocks.mocks import nmpcMockOptions

import time
import os
import io
import json
import requests
import tiktoken
from PIL import Image
from copy import deepcopy
from datetime import datetime
from streamlit import empty, session_state
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, AIMessage
from langchain.prompts.chat import SystemMessagePromptTemplate
from langchain.output_parsers import PydanticOutputParser
from langchain.callbacks.base import BaseCallbackHandler
import numpy as np
import openai
import ast
import astunparse
from pygments import highlight
from pygments.lexers.python import PythonLexer
from pygments.formatters import TerminalFormatter
import copy
import shapely
from shapely.geometry import *
from shapely.affinity import *
from prompts.prompts import prompt_tabletop_ui, prompt_parse_obj_name, prompt_parse_position, prompt_parse_question, prompt_transform_shape_pts, prompt_fgen
from db import Episode, Epoch
from config.config import SimulationConfig, RobotConfig
model_name = "gpt-4-0125-preview" # "gpt-3.5-turbo-instruct" # "gpt-4-0125-preview" # "davinci-002"  # "gpt-4"

episode = None

TOKEN_ENCODER = tiktoken.encoding_for_model("gpt-4")
global_log = ""
global_log_chat = ""

def append_to_chat_log(message:str):
  global global_log_chat
  global_log_chat += message + "\n\n"

def clear_global_log():
  global global_log
  global_log = ""

def clear_global_log_chat():
  global global_log_chat
  global_log_chat = ""

class Message:
  def __init__(self, text, base64_image=None, role="user"):
    self.role = role
    self.text = text
    self.base64_image = base64_image

  def to_dict(self):
    message = [{"type": "text", "text": self.text}]
    if self.base64_image:
      message.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{self.base64_image}", "detail": "high"}})
    return {"role": self.role, "content": message}

class StreamHandler(BaseCallbackHandler):

  def __init__(self, avatar:str, parser: PydanticOutputParser) -> None:
    super().__init__()
    self.avatar = avatar
    self.parser = parser

  def on_llm_start(self, serialized, prompts, **kwargs) -> None:
    """Run when LLM starts running."""
    self.text = ""
    self.container = empty()

  def on_llm_new_token(self, token: str, *, chunk, run_id, parent_run_id=None, **kwargs):
    super().on_llm_new_token(token, chunk=chunk, run_id=run_id, parent_run_id=parent_run_id, **kwargs)
    self.text += token
    self.container.write(self.text + "|")

  def on_llm_end(self, response, **kwargs):
    pretty_text = self.parser.parse(self.text).pretty_print()
    self.container.markdown(pretty_text, unsafe_allow_html=False)
    session_state.messages.append({"type": self.avatar, "content": pretty_text})

class LLM(AbstractLLM):

  def __init__(self, cfg: AbstractLLMConfig) -> None:
    super().__init__(cfg)

    # init messages
    self.messages = [Message(text=self.cfg.prompt, role="system")]
    # request headers
    self.headers = {
      "Content-Type": "application/json",
      "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}"
    }

  def reset(self):
    self.messages = [Message(text=self.cfg.prompt, role="system")]

  def run(self, user_message:str, base64_image=None, short_history=False) -> str:
    # add user message to chat history
    self.messages.append(Message(text=user_message, role="user", base64_image=base64_image))
    # select the last 2 user messages and the last assistant message
    selected_messages = [self.messages[0]] + [m for m in self.messages[-2:] if m.role!="system"] if short_history else self.messages
    # send request to OpenAI API
    payload = {
      "model": self.cfg.model_name,
      "messages": [m.to_dict() for m in selected_messages],
      "max_tokens": self.cfg.max_tokens,
      "response_format": {"type": "json_object"}
    }
    #print([m.text for m in selected_messages])
    response = requests.post("https://api.openai.com/v1/chat/completions", headers=self.headers, json=payload).json()
    # retrieve text response
    try:
      AI_response = response['choices'][0]['message']['content']
      self.messages.append(Message(text=AI_response, role="assistant"))
      print(f"\33[92m {AI_response} \033[0m \n")
      AI_response = json.loads(AI_response)
    except Exception as e:
      print(f"Error: {e}")
      AI_response = {"instruction": response['error']['message']}

    return AI_response


class LMP:

    def __init__(self, name, cfg, lmp_fgen, fixed_vars, variable_vars, db_sessionmaker, main_lmp=False):
        self._name = name
        self._cfg = cfg

        self._base_prompt = self._cfg['prompt_text']

        self._stop_tokens = list(self._cfg['stop'])

        self._lmp_fgen = lmp_fgen

        self._fixed_vars = fixed_vars
        self._variable_vars = variable_vars
        self.exec_hist = ''
        self.main_lmp = main_lmp

        self.sessionmaker = db_sessionmaker
    
    def store_epoch_db(self, episode_id, role, content, image_url):
      session = self.sessionmaker()
      
      # Find the last epoch number for this episode
      last_epoch = session.query(Epoch).filter_by(episode_id=episode_id).order_by(Epoch.time_step.desc()).first()
      if last_epoch is None:
          next_time_step = 1  # This is the first epoch for the episode
      else:
          next_time_step = last_epoch.time_step + 1
      
      # Create and insert the new epoch
      epoch = Epoch(episode_id=episode_id, time_step=next_time_step, role=role, content=content, image=image_url)
      session.add(epoch)
      session.commit()
      session.close()
      clear_global_log()
    
    def update_obs(self, obs):
        self._variable_vars['_update_obs'](obs)

    def clear_exec_hist(self):
        self.exec_hist = ''

    def reset(self):
        self.clear_exec_hist()

    def build_prompt(self, query, context=''):
        if len(self._variable_vars) > 0:
            variable_vars_imports_str = f"from utils import {', '.join(self._variable_vars.keys())}"
        else:
            variable_vars_imports_str = ''
        prompt = self._base_prompt.replace('{variable_vars_imports}', variable_vars_imports_str)

        if self._cfg['maintain_session']:
            prompt += f'\n{self.exec_hist}'

        if context != '':
            prompt += f'\n{context}'

        use_query = f'{self._cfg["query_prefix"]}{query}{self._cfg["query_suffix"]}'
        prompt += f'\n{use_query}'

        return prompt, use_query

    def __call__(self, query, episode_local, context='', **kwargs):
        global episode
        episode = episode_local

        prompt, use_query = self.build_prompt(query, context=context)
        print(f"[LMP] {prompt=}")

        while True:
            try:
                code_str = request_oai(prompt, model_name=self._cfg["engine"])
                # code_str = openai.Completion.create(
                #     prompt=prompt,
                #     stop=self._stop_tokens,
                #     temperature=self._cfg['temperature'],
                #     engine=self._cfg['engine'],
                #     max_tokens=self._cfg['max_tokens']
                # )['choices'][0]['text'].strip()
                log_msg = f"[LMP, Response] {code_str}"
                append_to_chat_log(log_msg)
                self.store_epoch_db(episode.id, "ai", log_msg, "")
                break
            except Exception as e:
                print(f'OpenAI API got err {e}')
                print('Retrying after 10s.')
                sleep(10)

        if self._cfg['include_context'] and context != '':
            to_exec = f'{context}\n{code_str}'
            to_log = f'{context}\n{use_query}\n{code_str}'
        else:
            to_exec = code_str
            to_log = f'{use_query}\n{to_exec}'

        to_log_pretty = highlight(to_log, PythonLexer(), TerminalFormatter())
        print(f'LMP {self._name} exec:\n\n{to_log_pretty}\n')

        new_fs = self._lmp_fgen.create_new_fs_from_code(code_str)
        self._variable_vars.update(new_fs)

        gvars = merge_dicts([self._fixed_vars, self._variable_vars])
        lvars = kwargs

        if not self._cfg['debug_mode']:
            exec_safe(to_exec, gvars, lvars)

        self.exec_hist += f'\n{to_exec}'

        if self._cfg['maintain_session']:
            self._variable_vars.update(lvars)

        # if self._cfg['has_return']:
        #     return lvars[self._cfg['return_val_name']]
        log = deepcopy(global_log_chat)
        if self.main_lmp:
            clear_global_log_chat()
        return log


class LMPFGen:

    def __init__(self, cfg, fixed_vars, variable_vars, sessionmaker):
        self._cfg = cfg

        self._stop_tokens = list(self._cfg['stop'])
        self._fixed_vars = fixed_vars
        self._variable_vars = variable_vars

        self._base_prompt = self._cfg['prompt_text']

        self.sessionmaker = sessionmaker
    
    def store_epoch_db(self, episode_id, role, content, image_url):
        session = self.sessionmaker()
        
        # Find the last epoch number for this episode
        last_epoch = session.query(Epoch).filter_by(episode_id=episode_id).order_by(Epoch.time_step.desc()).first()
        if last_epoch is None:
            next_time_step = 1  # This is the first epoch for the episode
        else:
            next_time_step = last_epoch.time_step + 1
        
        # Create and insert the new epoch
        epoch = Epoch(episode_id=episode_id, time_step=next_time_step, role=role, content=content, image=image_url)
        session.add(epoch)
        session.commit()
        session.close()
        clear_global_log()

    def create_f_from_sig(self, f_name, f_sig, other_vars=None, fix_bugs=False, return_src=False):
        print(f'Creating function: {f_sig}')

        use_query = f'{self._cfg["query_prefix"]}{f_sig}{self._cfg["query_suffix"]}'
        prompt = f'{self._base_prompt}\n{use_query}'
        print(f"[LMPFGen] {prompt=}")
        while True:
            try:
                f_src = request_oai(prompt, model_name=self._cfg["engine"])
                # f_src = openai.Completion.create(
                #     prompt=prompt, 
                #     stop=self._stop_tokens,
                #     temperature=self._cfg['temperature'],
                #     engine=self._cfg['engine'],
                #     max_tokens=self._cfg['max_tokens']
                # )['choices'][0]['text'].strip()
                log_msg = f"[LMPFGen, Response] {f_src}"
                append_to_chat_log(log_msg)
                self.store_epoch_db(episode.id, "ai", log_msg, "")
                break
            except Exception as e:
                print(f'OpenAI API got err {e}')
                print('Retrying after 10s.')
                sleep(10)

        if fix_bugs:
            f_src = openai.Edit.create(
                model='code-davinci-edit-001',
                input='# ' + f_src,
                temperature=0,
                instruction='Fix the bug if there is one. Improve readability. Keep same inputs and outputs. Only small changes. No comments.',
            )['choices'][0]['text'].strip()

        if other_vars is None:
            other_vars = {}
        gvars = merge_dicts([self._fixed_vars, self._variable_vars, other_vars])
        lvars = {}
        
        exec_safe(f_src, gvars, lvars)

        f = lvars[f_name]

        to_print = highlight(f'{use_query}\n{f_src}', PythonLexer(), TerminalFormatter())
        print(f'LMP FGEN created:\n\n{to_print}\n')

        if return_src:
            return f, f_src
        return f

    def create_new_fs_from_code(self, code_str, other_vars=None, fix_bugs=False, return_src=False):
        fs, f_assigns = {}, {}
        f_parser = FunctionParser(fs, f_assigns)
        f_parser.visit(ast.parse(code_str))
        for f_name, f_assign in f_assigns.items():
            if f_name in fs:
                fs[f_name] = f_assign

        if other_vars is None:
            other_vars = {}

        new_fs = {}
        srcs = {}
        for f_name, f_sig in fs.items():
            all_vars = merge_dicts([self._fixed_vars, self._variable_vars, new_fs, other_vars])
            if not var_exists(f_name, all_vars):
                f, f_src = self.create_f_from_sig(f_name, f_sig, new_fs, fix_bugs=fix_bugs, return_src=True)

                # recursively define child_fs in the function body if needed
                f_def_body = astunparse.unparse(ast.parse(f_src).body[0].body)
                child_fs, child_f_srcs = self.create_new_fs_from_code(
                    f_def_body, other_vars=all_vars, fix_bugs=fix_bugs, return_src=True
                )

                if len(child_fs) > 0:
                    new_fs.update(child_fs)
                    srcs.update(child_f_srcs)

                    # redefine parent f so newly created child_fs are in scope
                    gvars = merge_dicts([self._fixed_vars, self._variable_vars, new_fs, other_vars])
                    lvars = {}
                    
                    exec_safe(f_src, gvars, lvars)
                    
                    f = lvars[f_name]

                new_fs[f_name], srcs[f_name] = f, f_src

        if return_src:
            return new_fs, srcs
        return new_fs


class FunctionParser(ast.NodeTransformer):

    def __init__(self, fs, f_assigns):
      super().__init__()
      self._fs = fs
      self._f_assigns = f_assigns

    def visit_Call(self, node):
        self.generic_visit(node)
        if isinstance(node.func, ast.Name):
            f_sig = astunparse.unparse(node).strip()
            f_name = astunparse.unparse(node.func).strip()
            self._fs[f_name] = f_sig
        return node

    def visit_Assign(self, node):
        self.generic_visit(node)
        if isinstance(node.value, ast.Call):
            assign_str = astunparse.unparse(node).strip()
            f_name = astunparse.unparse(node.value.func).strip()
            self._f_assigns[f_name] = assign_str
        return node


def var_exists(name, all_vars):
    try:
        eval(name, all_vars)
    except:
        exists = False
    else:
        exists = True
    return exists


def merge_dicts(dicts):
    return {
        k : v 
        for d in dicts
        for k, v in d.items()
    }
    

def exec_safe(code_str, gvars=None, lvars=None):
    banned_phrases = ['import', '__']
    for phrase in banned_phrases:
        assert phrase not in code_str
  
    if gvars is None:
        gvars = {}
    if lvars is None:
        lvars = {}
    empty_fn = lambda *args, **kwargs: None
    custom_gvars = merge_dicts([
        gvars,
        {'exec': empty_fn, 'eval': empty_fn}
    ])
    exec(code_str, custom_gvars, lvars)
  
class LMP_wrapper():
  """Where all the primitives are defined. This is the interface between the LMPs and the environment."""

  def __init__(self, env, cfg, mpc, sessionmaker, task_name, render=False):
    self._cfg = cfg
    self.cfg = SimulationConfig()
    self.robot_cfg = RobotConfig()
    self.env = env
    self.object_names = list(self._cfg['env']['init_objs'])
    
    self._min_xy = np.array(self._cfg['env']['coords']['bottom_left'])
    self._max_xy = np.array(self._cfg['env']['coords']['top_right'])
    self._range_xy = self._max_xy - self._min_xy

    self._table_z = self._cfg['env']['coords']['table_z']
    self.render = render

    self.mpc = mpc
    self.gripper = 1.  # open
    self.t = 0

    self.sessionmaker = sessionmaker
    self.task_name = task_name

  def store_epoch_db(self, episode_id, role, content, image_url):
    session = self.sessionmaker()
    
    # Find the last epoch number for this episode
    last_epoch = session.query(Epoch).filter_by(episode_id=episode_id).order_by(Epoch.time_step.desc()).first()
    if last_epoch is None:
        next_time_step = 1  # This is the first epoch for the episode
    else:
        next_time_step = last_epoch.time_step + 1
    
    # Create and insert the new epoch
    epoch = Epoch(episode_id=episode_id, time_step=next_time_step, role=role, content=content, image=image_url)
    session.add(epoch)
    session.commit()
    session.close()
    clear_global_log()

  def _open_gripper(self):
    self.gripper = -0.01
    self.gripper_timer = 0
    while self.gripper < 1.:
      if self.gripper<0.9 and self.gripper_timer>self.robot_cfg.open_gripper_time: 
        self.gripper = 1.
      else:
        self.gripper_timer += 1
      self.obs, _, done, _ = self.env.step([np.array([0., 0., 0., 0., 0., 0., self.gripper])])
    

  def _close_gripper(self):
    self.gripper = -0.02
    for _ in range(15):
      self.obs, _, done, _ = self.env.step([np.array([0., 0., 0., 0., 0., 0., self.gripper])])
  
  def _update_obs(self, obs):
    self.obs = obs

  # def is_obj_visible(self, obj_name):
  #   return obj_name in self.object_names

  def get_obj_names(self):
    return self.object_names[::]

  def denormalize_xy(self, pos_normalized):
    return pos_normalized * self._range_xy + self._min_xy

  # def get_corner_positions(self):
  #   unit_square = box(0, 0, 1, 1)
  #   normalized_corners = np.array(list(unit_square.exterior.coords))[:4]
  #   corners = np.array(([self.denormalize_xy(corner) for corner in normalized_corners]))
  #   return corners

  def get_side_positions(self):
    side_xs = np.array([0, 0.5, 0.5, 1])
    side_ys = np.array([0.5, 0, 1, 0.5])
    normalized_side_positions = np.c_[side_xs, side_ys]
    side_positions = np.array(([self.denormalize_xy(corner) for corner in normalized_side_positions]))
    return side_positions

  def get_obj_pos(self, obj_name):
    # return the xyz position of the object in robot base frame
    return list(self.obs[obj_name]["position"])

  # def get_obj_position_np(self, obj_name):
  #   return self.get_pos(obj_name)

  # def get_bbox(self, obj_name):
  #   # return the axis-aligned object bounding box in robot base frame (not in pixels)
  #   # the format is (min_x, min_y, max_x, max_y)
  #   bbox = self.env.get_bounding_box(obj_name)
  #   return bbox

  # def get_color(self, obj_name):
  #   for color, rgb in COLORS.items():
  #     if color in obj_name:
  #       return rgb

  def _upload_image(self, rgba_image:np.ndarray) -> str:
    # Convert the NumPy array to a PIL Image object
    image = Image.fromarray(rgba_image, 'RGBA')
    # Convert the PIL Image object to a byte stream
    byte_stream = io.BytesIO()
    image.save(byte_stream, format='PNG')  # You can change PNG to JPEG if preferred
    byte_stream.seek(0)  # Seek to the start of the stream

    # # Imgur API details
    # client_id = 'c978542bde3df32'  # Replace with your Imgur Client ID
    # headers = {'Authorization': f'Client-ID {client_id}'}

    # # Prepare the data for the request
    # data = {'image': byte_stream.read()}

    # # Make the POST request to upload the image
    # response = requests.post('https://api.imgur.com/3/upload', headers=headers, files=data)

    # if response.status_code == 200:
    #     # Return the image link
    #     return response.json()['data']['link']
    # else:
    
    image_path = f'data/images/cap/{self.task_name}/{episode.id}_{datetime.now().strftime("%d-%m-%Y_%H:%M:%S")}.png'
    image.save(image_path, 'PNG')
    return image_path
      
  def _retrieve_image(self) -> np.ndarray:
    frame_np = np.array(self.env.render("rgb_array", 
                                        width=self.cfg.frame_width, height=self.cfg.frame_height,
                                        target_position=self.cfg.frame_target_position,
                                        distance=self.cfg.frame_distance,
                                        yaw=self.cfg.frame_yaw,
                                        pitch=self.cfg.frame_pitch))
    frame_np = frame_np.reshape(self.cfg.frame_width, self.cfg.frame_height, 4).astype(np.uint8)

    return frame_np
  
  def _is_robot_busy(self):
    print(f"{self.mpc.cost=}")
    print(f"{self.mpc.prev_cost=}")
    c1 = self.mpc.prev_cost - self.mpc.cost <= self.robot_cfg.COST_DIIFF_THRESHOLD if self.mpc.prev_cost is not None else False
    c2 = self.mpc.cost <= self.robot_cfg.COST_THRESHOLD
    c3 = time.time()-self.t_prev_task>=self.robot_cfg.TIME_THRESHOLD
    print(f"Robot busy: {c1=}, {c2=}, {c3=}")
    return not (c1 or c2 or c3)

  def run_mpc(self, optimization):
    global episode
    self.mpc.init_states(self.obs, self.t, False)
    self.mpc.setup_controller(optimization)
    self.t_prev_task = time.time()
    print(f"{optimization=}")
    while self._is_robot_busy():
      self.mpc.init_states(self.obs, self.t, False)
      action = []
      control: List[np.ndarray] = self.mpc.step()
      for u in control:
        action.append(np.hstack((u, self.gripper)))
      
      trajectory = self.mpc.retrieve_trajectory()
      self.env.visualize_trajectory(trajectory)
      self.obs, _, done, _ = self.env.step(action)
    
    image = self._retrieve_image()
    image_url = self._upload_image(image)
    self.store_epoch_db(episode.id, "ai", deepcopy(global_log), image_url)

  def move_obj_to_pos(self, obj_name, target_pos):
      # move the object to the desired xyz position
      pick_pos = self.get_obj_pos(obj_name) if isinstance(obj_name, str) else obj_name
      place_pos = list(np.array(target_pos))
      self._open_gripper()

      above_pick_pos = list(np.array(pick_pos) + np.array([0, 0, 0.1]))
      optimization = {
        "objective": f"ca.norm_2(x - np.array({above_pick_pos}))**2",
        "equality_constraints":[],
        "inequality_constraints":[]
      }
      self.run_mpc(optimization)

      optimization = {
        "objective": f"ca.norm_2(x - np.array({pick_pos}))**2",
        "equality_constraints":[],
        "inequality_constraints":[]
      }
      self.run_mpc(optimization)

      self._close_gripper()

      optimization = {
        "objective": f"ca.norm_2(x - np.array({above_pick_pos}))**2",
        "equality_constraints":[],
        "inequality_constraints":[]
      }
      self.run_mpc(optimization)


      above_place_pos = list(np.array(place_pos) + np.array([0, 0, 0.1]))
      optimization = {
        "objective": f"ca.norm_2(x - np.array({above_place_pos}))**2",
        "equality_constraints":[],
        "inequality_constraints":[]
      }
      self.run_mpc(optimization)

      optimization = {
        "objective": f"ca.norm_2(x - np.array({place_pos}))**2",
        "equality_constraints":[],
        "inequality_constraints":[]
      }
      self.run_mpc(optimization)

      self._open_gripper()

      above_place_pos = list(np.array(place_pos) + np.array([0, 0, 0.1]))
      optimization = {
        "objective": f"ca.norm_2(x - np.array({above_place_pos}))**2",
        "equality_constraints":[],
        "inequality_constraints":[]
      }
      self.run_mpc(optimization)

  def put_first_on_second(self, arg1, arg2):
    # put the object with obj_name on top of target
    print(f"Called put_first_on_second with {arg1=} and {arg2=}")
    place_pos = self.get_obj_pos(arg2) if isinstance(arg2, str) else arg2
    place_pos = list(np.array(place_pos) + np.array([0, 0, 0.04]))
    self.move_obj_to_pos(arg1, place_pos)
    print(f"put_first_on_second done DONE")


  def get_robot_pos(self):
    # return robot end-effector xyz position in robot base frame
    print(f"robot position: {self.obs['robot'][:3]}")
    return self.obs['robot'][:3]

  def goto_pos(self, position_xy):
    # move the robot end-effector to the desired xy position while maintaining same z
    ee_xyz = self.get_robot_pos()
    position_xyz = np.concatenate([position_xy, ee_xyz[-1]])
    while np.linalg.norm(position_xyz - ee_xyz) > 0.01:
      # TODO implement in env
      self.env.movep(position_xyz)
      # self.env.step_sim_and_render()
      # ee_xyz = self.env.get_ee_pos()

  def follow_traj(self, traj):
    # traj is a list of xy positions
    for pos in traj:
      self.goto_pos(pos)
  
  def get_corner_positions(self):
    normalized_corners = np.array([
        [0, 1],
        [1, 1],
        [0, 0],
        [1, 0]
    ])
    return np.array(([self.denormalize_xy(corner) for corner in normalized_corners]))

  def get_side_positions(self):
    normalized_sides = np.array([
        [0.5, 1],
        [1, 0.5],
        [0.5, 0],
        [0, 0.5]
    ])
    return np.array(([self.denormalize_xy(side) for side in normalized_sides]))

  def get_corner_name(self, pos):
    corner_positions = self.get_corner_positions()
    corner_idx = np.argmin(np.linalg.norm(corner_positions - pos, axis=1))
    return ['top left corner', 'top right corner', 'bottom left corner', 'botom right corner'][corner_idx]

  def get_side_name(self, pos):
    side_positions = self.get_side_positions()
    side_idx = np.argmin(np.linalg.norm(side_positions - pos, axis=1))
    return ['top side', 'right side', 'bottom side', 'left side'][side_idx]
  
# class LMP_wrapper_mock:
#    def __init__(self, env, cfg) -> None:
#       pass
   
#    def get_obj_pos(self, obj_name):
#       print("Called get_obj_pos")
#       return np.array([0, 0, 0])
   
#    def get_obj_names(self,):
#       print("Called get_obj_names")
#       return ["red_cube", "green_cube"]
   
#    def put_first_on_second(self, obj_a, obj_b):
#       print("Called put_first_on_second")
      

cfg_tabletop = {
  'lmps': {
    'tabletop_ui': {
      'prompt_text': prompt_tabletop_ui,
      'engine': model_name,
      'max_tokens': 512,
      'temperature': 0,
      'query_prefix': '# ',
      'query_suffix': '.',
      'stop': ['#', 'objects = ['],
      'maintain_session': True,
      'debug_mode': False,
      'include_context': True,
      'has_return': False,
      'return_val_name': 'ret_val',
    },
    'parse_obj_name': {
      'prompt_text': prompt_parse_obj_name,
      'engine': model_name,
      'max_tokens': 512,
      'temperature': 0,
      'query_prefix': '# ',
      'query_suffix': '.',
      'stop': ['#', 'objects = ['],
      'maintain_session': False,
      'debug_mode': False,
      'include_context': True,
      'has_return': True,
      'return_val_name': 'ret_val',
    },
    'parse_position': {
      'prompt_text': prompt_parse_position,
      'engine': model_name,
      'max_tokens': 512,
      'temperature': 0,
      'query_prefix': '# ',
      'query_suffix': '.',
      'stop': ['#'],
      'maintain_session': False,
      'debug_mode': False,
      'include_context': True,
      'has_return': True,
      'return_val_name': 'ret_val',
    },
    'parse_question': {
      'prompt_text': prompt_parse_question,
      'engine': model_name,
      'max_tokens': 512,
      'temperature': 0,
      'query_prefix': '# ',
      'query_suffix': '.',
      'stop': ['#', 'objects = ['],
      'maintain_session': False,
      'debug_mode': False,
      'include_context': True,
      'has_return': True,
      'return_val_name': 'ret_val',
    },
    'transform_shape_pts': {
      'prompt_text': prompt_transform_shape_pts,
      'engine': model_name,
      'max_tokens': 512,
      'temperature': 0,
      'query_prefix': '# ',
      'query_suffix': '.',
      'stop': ['#'],
      'maintain_session': False,
      'debug_mode': False,
      'include_context': True,
      'has_return': True,
      'return_val_name': 'new_shape_pts',
    },
    'fgen': {
      'prompt_text': prompt_fgen,
      'engine': model_name,
      'max_tokens': 512,
      'temperature': 0,
      'query_prefix': '# define function: ',
      'query_suffix': '.',
      'stop': ['# define', '# example'],
      'maintain_session': False,
      'debug_mode': False,
      'include_context': True,
    }
  }
}

lmp_tabletop_coords = {
        'top_left':     (-0.3 + 0.05, -0.2 - 0.05),
        'top_side':     (0,           -0.2 - 0.05),
        'top_right':    (0.3 - 0.05,  -0.2 - 0.05),
        'left_side':    (-0.3 + 0.05, -0.5,      ),
        'middle':       (0,           -0.5,      ),
        'right_side':   (0.3 - 0.05,  -0.5,      ),
        'bottom_left':  (-0.3 + 0.05, -0.8 + 0.05),
        'bottom_side':  (0,           -0.8 + 0.05),
        'bottom_right': (0.3 - 0.05,  -0.8 + 0.05),
        'table_z':       0.0,
      }


def setup_LMP(env, cfg_tabletop, mpc, db_sessionmaker, task_name):
  # LMP env wrapper
  cfg_tabletop = copy.deepcopy(cfg_tabletop)
  cfg_tabletop['env'] = dict()
  cfg_tabletop['env']['init_objs'] = list([obj['name'] for obj in env.objects_info])
  cfg_tabletop['env']['coords'] = lmp_tabletop_coords
  LMP_env = LMP_wrapper(env, cfg_tabletop, mpc, db_sessionmaker, task_name)
  # LMP_env = LMP_wrapper_mock(env, cfg_tabletop)

  # creating APIs that the LMPs can interact with
  fixed_vars = {
      'np': np
  }
  fixed_vars.update({
      name: eval(name)
      for name in shapely.geometry.__all__ + shapely.affinity.__all__
  })
  # variable_vars = {
  #     k: getattr(LMP_env, k)
  #     for k in [
  #         'get_obj_pos', 'get_robot_pos', 'denormalize_xy',
  #         'put_first_on_second', 'get_obj_names',
  #         'get_corner_name', 'get_side_name',
  #     ]
  # }
  variable_vars = {
      k: getattr(LMP_env, k)
      for k in [
          'get_obj_pos', 'get_robot_pos', '_update_obs', 'put_first_on_second', 'move_obj_to_pos', 'get_obj_names',
          
      ]
  }
  variable_vars['say'] = lambda msg: print(f'robot says: {msg}')

  # creating the function-generating LMP
  lmp_fgen = LMPFGen(cfg_tabletop['lmps']['fgen'], fixed_vars, variable_vars, db_sessionmaker)

  # creating other low-level LMPs
  variable_vars.update({
      k: LMP(k, cfg_tabletop['lmps'][k], lmp_fgen, fixed_vars, variable_vars, db_sessionmaker)
      for k in ['parse_obj_name', 'parse_position', 'parse_question', 'transform_shape_pts']
  })

  # creating the LMP that deals w/ high-level language commands
  lmp_tabletop_ui = LMP(
      'tabletop_ui', cfg_tabletop['lmps']['tabletop_ui'], lmp_fgen, fixed_vars, variable_vars, db_sessionmaker, main_lmp=True
  )

  return lmp_tabletop_ui

def request_oai(message, model_name, max_tokens=512):
  if model_name == "davinci-002" or model_name == "gpt-3.5-turbo-instruct":
    return request_oai_legacy(message, model_name, max_tokens)
  else:
    return request_oai_chat(message, model_name, max_tokens)

def request_oai_chat(message, model_name, max_tokens=512):
    payload = {
      "model": model_name,
      "messages": [{"role": "user", "content": message}],
      "max_tokens": max_tokens,
    }
    headers = {
      "Content-Type": "application/json",
      "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}"
    }
    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload).json()
    AI_response = response['choices'][0]['message']['content']
    print(f"\33[92m {AI_response} \033[0m \n")
    return AI_response

def request_oai_legacy(message, model_name, max_tokens=512):
    payload = {
      "model": model_name,
      "prompt": message,
      "max_tokens": max_tokens,
    }
    headers = {
      "Content-Type": "application/json",
      "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}"
    }
    response = requests.post("https://api.openai.com/v1/completions", headers=headers, json=payload).json()
    AI_response = response['choices'][0]['text']
    print(f"\33[92m {AI_response} \033[0m \n")
    return AI_response
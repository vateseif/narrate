from time import sleep
from typing import List, Optional
from core import AbstractLLM, AbstractLLMConfig
from mocks.mocks import nmpcMockOptions

import os
import json
import requests
import tiktoken
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
model_name = "gpt-4"


TOKEN_ENCODER = tiktoken.encoding_for_model("gpt-4")

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
    self.messages = []
    # load prompt
    self.messages.append(Message(text=self.cfg.prompt, role="system"))
    # request headers
    self.headers = {
      "Content-Type": "application/json",
      "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}"
    }

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

    def __init__(self, name, cfg, lmp_fgen, fixed_vars, variable_vars):
        self._name = name
        self._cfg = cfg

        self._base_prompt = self._cfg['prompt_text']

        self._stop_tokens = list(self._cfg['stop'])

        self._lmp_fgen = lmp_fgen

        self._fixed_vars = fixed_vars
        self._variable_vars = variable_vars
        self.exec_hist = ''
    
    def update_obs(self, obs):
        self._variable_vars['_update_obs'](obs)

    def clear_exec_hist(self):
        self.exec_hist = ''

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

    def __call__(self, query, context='', **kwargs):
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

        if self._cfg['has_return']:
            return lvars[self._cfg['return_val_name']]


class LMPFGen:

    def __init__(self, cfg, fixed_vars, variable_vars):
        self._cfg = cfg

        self._stop_tokens = list(self._cfg['stop'])
        self._fixed_vars = fixed_vars
        self._variable_vars = variable_vars

        self._base_prompt = self._cfg['prompt_text']

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

  def __init__(self, env, cfg, render=False):
    self._cfg = cfg
    self.env = env
    self.object_names = list(self._cfg['env']['init_objs'])
    
    self._min_xy = np.array(self._cfg['env']['coords']['bottom_left'])
    self._max_xy = np.array(self._cfg['env']['coords']['top_right'])
    self._range_xy = self._max_xy - self._min_xy

    self._table_z = self._cfg['env']['coords']['table_z']
    self.render = render
  
  def _update_obs(self, obs):
    self.obs = obs
    print(f"[LMP_wrapper] Updated obs: {self.obs}")

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
    print(f"self.env.objects_info: {self.env.objects_info}")
    return self.obs[obj_name]["position"]

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

  def put_first_on_second(self, arg1, arg2):
    # put the object with obj_name on top of target
    # target can either be another object name, or it can be an x-y position in robot base frame
    pick_pos = self.get_obj_pos(arg1) if isinstance(arg1, str) else arg1
    place_pos = self.get_obj_pos(arg2) if isinstance(arg2, str) else arg2
    # TODO implement in env
    self.env.step_pick_place(action={'pick': pick_pos, 'place': place_pos})

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
  
  def open_gripper(self):
    # TODO implement in env
    self.env.open_gripper()
  
  def close_gripper(self):
    # TODO implement in env
    self.env.close_gripper()

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


def setup_LMP(env, cfg_tabletop):
  # LMP env wrapper
  cfg_tabletop = copy.deepcopy(cfg_tabletop)
  cfg_tabletop['env'] = dict()
  cfg_tabletop['env']['init_objs'] = list([obj['name'] for obj in env.objects_info])
  cfg_tabletop['env']['coords'] = lmp_tabletop_coords
  LMP_env = LMP_wrapper(env, cfg_tabletop)
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
          'get_obj_pos', 'get_robot_pos', '_update_obs',
      ]
  }
  variable_vars['say'] = lambda msg: print(f'robot says: {msg}')
  print(f"variable_vars: {variable_vars}")

  # creating the function-generating LMP
  lmp_fgen = LMPFGen(cfg_tabletop['lmps']['fgen'], fixed_vars, variable_vars)

  # creating other low-level LMPs
  variable_vars.update({
      k: LMP(k, cfg_tabletop['lmps'][k], lmp_fgen, fixed_vars, variable_vars)
      for k in ['parse_obj_name', 'parse_position', 'parse_question', 'transform_shape_pts']
  })

  # creating the LMP that deals w/ high-level language commands
  lmp_tabletop_ui = LMP(
      'tabletop_ui', cfg_tabletop['lmps']['tabletop_ui'], lmp_fgen, fixed_vars, variable_vars
  )

  return lmp_tabletop_ui

def request_oai(message, model_name, max_tokens=512):
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
    print(f"{response=}")
    AI_response = response['choices'][0]['message']['content']
    return AI_response
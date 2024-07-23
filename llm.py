from time import time
from core import AbstractLLM, AbstractLLMConfig

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
    self.messages = [Message(text=self.cfg.prompt, role="system")]
    # request headers
    self.headers = {
      "Content-Type": "application/json",
      "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}"
    }

  def reset(self):
    self.messages = [Message(text=self.cfg.prompt, role="system")]

  def run(self, user_message:str, base64_image=None, short_history=False) -> dict:
    # add user message to chat history
    self.messages.append(Message(text=user_message, role="user", base64_image=base64_image))
    # select the last 2 user messages and the last assistant message
    selected_messages = [self.messages[0]] + [m for m in self.messages[-1:] if m.role!="system"] if short_history else self.messages
    # send request to OpenAI API
    payload = {
      "model": self.cfg.model_name,
      "messages": [m.to_dict() for m in selected_messages],
      "max_tokens": self.cfg.max_tokens,
      "response_format": {"type": "json_object"}
    }
    #print([m.text for m in selected_messages])
    t0 = time()
    response = requests.post("https://api.openai.com/v1/chat/completions", headers=self.headers, json=payload).json()
    solve_time = time() - t0
    # retrieve text response
    try:
      AI_response = response['choices'][0]['message']['content']
      self.messages.append(Message(text=AI_response, role="assistant"))
      AI_response = json.loads(AI_response)
    except Exception as e:
      print(f"Error: {e}")
      AI_response = {"instruction": response['error']['message']}

    AI_response["solve_time"] = solve_time
    return AI_response


  


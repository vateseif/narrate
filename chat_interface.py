import os
import tiktoken
import streamlit as st
from time import sleep
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, AIMessage
from langchain.callbacks.base import BaseCallbackHandler

class StreamHandler(BaseCallbackHandler):

  def on_llm_start(self, serialized, prompts, **kwargs) -> None:
    """Run when LLM starts running."""
    self.text = ""
    self.container = st.empty()

  def on_llm_new_token(self, token: str, **kwargs) -> None:
    self.text += token
    self.container.markdown(self.text + "▌")

  def on_llm_end(self, response, **kwargs):
    self.container.markdown(self.text)

def simulate_stream(text:str):
  """ Function used to simulate stream in case of harcoded responses """
  placeholder = st.empty()
  # Simulate stream of response with milliseconds delay
  partial_text = ""
  for chunk in enc.decode_batch([[x] for x in enc.encode(text)]):
      partial_text += chunk
      sleep(0.05)
      # Add a blinking cursor to simulate typing
      placeholder.markdown(partial_text + "▌")
  placeholder.markdown(text)
  # return AI message
  return AIMessage(content=text)

enc = tiktoken.encoding_for_model("gpt-4")
os.environ["OPENAI_API_KEY"] = open('keys/gpt4.key', 'r').readline().rstrip()
chat = ChatOpenAI(streaming=True, callbacks=[StreamHandler()], temperature=0)

st.title("Language to Optimization")


# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message.type):
        st.markdown(message.content)

# Accept user input
if prompt := st.chat_input("What is up?"):
    # Add user message to chat history
    st.session_state.messages.append(HumanMessage(content=prompt))
    # Display user message in chat message container
    with st.chat_message("human"):
        st.markdown(prompt)

    # Display assistant response in chat message container
    with st.chat_message("ai"):
      #response = chat(st.session_state.messages)
      ai_message = simulate_stream("")
      st.session_state.messages.append(ai_message)



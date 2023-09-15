import streamlit as st
from time import sleep
from langchain.schema import HumanMessage

from simulation import Simulation
from llm import simulate_stream

# Init streamlit page
st.title("Language to Optimization")

# Create sidebar
st.sidebar.title("Choose model")

# Add a sidebar radio button to select the message type
model = st.sidebar.radio("Select the model to talk to", ["Task Planner", "Optimization Designer"])

# init the avatars for the message icons
avatars = {"human":None, "OD":"images/wall-e.png", "TP":"images/eve.png"}

# init robot simulation
if "sim" not in st.session_state:
  # init simulator
  st.session_state.sim = Simulation()
  st.session_state.sim.run()
  # init state machine state:
  # 0 = There's no plan the OD can execute
  # 1 = There is a plan the TP can execute. A button pops up to allow the user to execute the plan
  # 2 = Trigger the execution of the plan
  st.session_state.stage = 0

def set_state(i):
  # Function to update the state machine stage
  st.session_state.stage = i

# Initialize chat history
if "messages" not in st.session_state:
  st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
  message_type = message["type"]
  with st.chat_message(message_type, avatar=avatars[message_type]):
    st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("What should the robot do?"):
  # Add user message to chat history
  st.session_state.messages.append({"type": "human", "content":prompt})
  # Display user message in chat message container
  with st.chat_message("human"):
    st.markdown(prompt)

  # Display assistant response in chat message container
  if model == "Task Planner":
    with st.chat_message("TP", avatar=avatars["TP"]):
      st.session_state.sim.create_plan(prompt, solve=False) 
      st.session_state.stage = 1
  elif model == "Optimization Designer":
    with st.chat_message("OD", avatar=avatars["OD"]):
      st.session_state.sim._solve_task(prompt)

if st.session_state.stage == 1:
  st.button('Execute plan', on_click=set_state, args=[2])

if st.session_state.stage == 2:
  with st.chat_message("OD", avatar=avatars["OD"]):
    st.session_state.sim.execute_plan()
  st.session_state.stage = 0

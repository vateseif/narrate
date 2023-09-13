import os
import streamlit as st
from time import sleep
from langchain.schema import HumanMessage

from simulation import Simulation


# Init streamlit page
st.title("Language to Optimization")

# Create sidebar
#st.sidebar.title("Message Type")

# Add a sidebar radio button to select the message type
#model = st.sidebar.radio("Select the model to talk to", ["Task Planner", "Optimization Designer"])

# init robot simulation
if "sim" not in st.session_state:
  st.session_state.sim = Simulation()
  st.session_state.sim.run()

# Initialize chat history
if "messages" not in st.session_state:
  st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
  with st.chat_message(message.type):
    st.markdown(message.content)

# Accept user input
if prompt := st.chat_input("What should the robot do?"):
  # Add user message to chat history
  st.session_state.messages.append(HumanMessage(content=prompt))
  # Display user message in chat message container
  with st.chat_message("human"):
    st.markdown(prompt)

  # Display assistant response in chat message container
  with st.chat_message("ai"):
    #if model == "Task Planner":
    st.session_state.sim.create_plan(prompt, solve=True) 
    #elif model == "Optimization Designer":
    #  st.session_state.sim._solve_task(prompt)



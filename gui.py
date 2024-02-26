import json
import requests
import streamlit as st
from time import sleep

# Init streamlit page
st.title("Language to Optimization")

# Create sidebar
st.sidebar.title("Choose model")

# Add a sidebar radio button to select the message type
model = st.sidebar.radio("Select the model to talk to", ["Task Planner", "Optimization Designer"])

# init the avatars for the message icons
avatars = {"human":"images/seif_avatar.jpeg", "OD":"images/wall-e.png", "TP":"images/eve.png"}

# base url to reach sim
base_url = 'http://localhost:8080/'

# init robot simulation
if "stage" not in st.session_state:
	# init state machine state:
	# 0 = There's no plan the OD can execute
	# 1 = There is a plan the TP can execute. A button pops up to allow the user to execute the plan
	# 2 = Trigger the execution of the plan
	st.session_state.stage = 0
	# init state machine state:
	# 0 = You can press start to start recording frames
	# 1 = You can press stop to save the recording or cacel the recording
	# 2 = Saves the recording and stops saving frames
	# 3 = cancels the recording and stops saving frames
	st.session_state.recording = 0
	# init number of tasks solved from a plan
	st.session_state.task = None

def set_state(i):
	# Function to update the state machine stage
	st.session_state.stage = i

def set_recording_state(i):
	# Function to update the recording state machine stage
	st.session_state.recording = i

def append_message(message:dict):
	# Function to append a message to the chat history
	message_type = message["type"]
	if message_type == "image":
		with st.chat_message("human", avatar=avatars["human"]):
			st.image(message["content"], width=400, caption="Current scene")
	else:
		with st.chat_message(message_type, avatar=avatars[message_type]):
			st.markdown(message["content"])

# Initialize chat history
if "messages" not in st.session_state:
	st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
	append_message(message)

# Accept user input
if prompt := st.chat_input("What should the robot do?"):
	# Add user message to chat history
	st.session_state.messages.append({"type": "human", "content":prompt})
	# Display user message in chat message container
	with st.chat_message("human", avatar=avatars["human"]):
		st.markdown(prompt)

	# Display assistant response in chat message container
	if model == "Task Planner":
		response = requests.post(base_url+'make_plan', json={"content": prompt}).json()
		st.session_state.messages += response
		for m in response: append_message(m)
		#st.session_state.task = response[-2]["content"]
		set_state(1)
	elif model == "Optimization Designer":
		response = requests.post(base_url+'solve_task', json={"content": prompt}).json()
		st.session_state.messages += response
		for m in response: append_message(m)

if st.session_state.stage == 1:
	st.button(f'Execute Plan', on_click=set_state, args=[2])

if st.session_state.stage == 2:
	st.button(f'Stop Plan', on_click=set_state, args=[0])
	response = requests.get(base_url+'next_task').json()
	if response[-1]['content'] == "finished":
		set_state(0)
	elif response[-1]['content'] is not None:
		for m in response: append_message(m)
		st.session_state.messages += response
	sleep(3)
	st.rerun()
	
# Reset the simulation
st.sidebar.button('Reset', on_click=requests.get, args=[base_url+'reset'])

# Close the simulation
st.sidebar.button('Close', on_click=requests.get, args=[base_url+'close'])

if st.session_state.recording == 0:
	st.sidebar.button('Start recording', on_click=set_recording_state, args=[1])

if st.session_state.recording == 1:
	_ = requests.get(base_url+'start_recording')
	st.sidebar.button('Stop recording', on_click=set_recording_state, args=[2])
	st.sidebar.button('Cancel recording', on_click=set_recording_state, args=[3])

if st.session_state.recording == 2:
	_ = requests.get(base_url+'save_recording')
	set_recording_state(0)

if st.session_state.recording == 3:
	_ = requests.get(base_url+'cancel_recording')
	set_recording_state(0)
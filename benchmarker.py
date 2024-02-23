import json
import numpy as np
import streamlit as st
from sqlalchemy import func
import matplotlib.pyplot as plt

from db import Session, Epoch, Episode

# Init db session
session = Session()	

# Init streamlit page
st.title("Language to Control")

# init the avatars for the message icons
avatars = {"human":"images/seif_avatar.jpeg", "OD":"images/wall-e.png", "TP":"images/eve.png"}

def plot_trajectory(episode_id):
	ep = session.query(Episode).filter_by(id=episode_id).first()
	try:
		state_trajectory = json.loads(ep.state_trajectories)
	except:
		st.write("No state trajectory available for this episode.")
		return
	state_trajectory = [np.array(s) for s in state_trajectory]

	efficiency = sum([np.linalg.norm(state_trajectory[i+1][:3] - state_trajectory[i][:3]) for i in range(len(state_trajectory)-1)])
	# 3d plot of the trajectory
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	ax.set_title("3D trajectory. Cumulative travel = {:.2f}m".format(efficiency))
	ax.plot([s[0] for s in state_trajectory], [-s[1] for s in state_trajectory], [s[2] for s in state_trajectory])

	# plot start and end
	ax.scatter(state_trajectory[0][0], -state_trajectory[0][1], state_trajectory[0][2], c='r', marker='o')
	ax.scatter(state_trajectory[-1][0], -state_trajectory[-1][1], state_trajectory[-1][2], c='g', marker='o')
	st.pyplot(fig)

def plot_mpc_times(episode_id):
	ep = session.query(Episode).filter_by(id=episode_id).first()
	try:
		mpc_times = json.loads(ep.mpc_solve_times)
	except:
		st.write("No MPC solve times available for this episode.")
		return
	fig = plt.figure()
	plt.title("MPC solve times")
	plt.plot(mpc_times)
	st.pyplot(fig)


def get_task_type(episode_id):	
	instruction = session.query(Epoch).filter_by(episode_id=episode_id).order_by(Epoch.time_step.asc()).first().content
	if "stack" in instruction:
		return f"{episode_id} (stack)"
	elif "L" in instruction:
		return f"{episode_id} (L)"
	else:
		return f"{episode_id} (pyramid)"

def append_message(message:dict):
	# Function to append a message to the chat history
	message_type = message["type"]
	if message_type == "image":
		with st.chat_message("human", avatar=avatars["human"]):
			st.image(message["content"], width=400, caption="Current scene")
	else:
		with st.chat_message(message_type, avatar=avatars[message_type]):
			st.markdown(message["content"])

	
# count number of episodes in db
n_episodes = session.query(func.count(Episode.id)).scalar()  # Assuming 'id' is a primary key column
episodes = [i for i in range(1, n_episodes+1)]
episode_id = st.selectbox('Select an Episode', options=episodes, format_func=get_task_type)


epochs = session.query(Epoch).filter_by(episode_id=episode_id).order_by(Epoch.time_step.desc()).all()
st.session_state.messages = []
for e in epochs:
	if e.image != "":
		st.session_state.messages += [{'type':'image', 'content':e.image}, {"type":e.role, 'content':e.content.replace("#","\n")}]
	else:
		st.session_state.messages += [{"type":e.role, 'content':e.content.replace("#","\n")}]


# read user query
append_message(st.session_state.messages[-1])
# Visualize last image to see if task completed succesfully
append_message(st.session_state.messages[0])

# plot 3D trajectory
plot_trajectory(episode_id)

# plot MPC solve times
plot_mpc_times(episode_id)

# close db session
session.close()

# Display chat messages from history on app rerun
for message in st.session_state.messages:
	append_message(message)

	


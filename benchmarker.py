import json
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from db import Epoch, Episode, Base

import os
wd = os.getcwd()


method = st.selectbox('Select a method', options=["ours", "ours_objective", "cap"])
task = st.selectbox('Select a task', options=["stack", "pyramid", "L", "CleanPlate", "Sponge"])


if method == "ours":
	if task in ["stack", "pyramid", "L"]:
		db_name = f'data/{method}/DBs/cubes.db'
	else:
		db_name = f'data/{method}/DBs/{task}.db'	
elif method == "ours_objective":
	if task == "CleanPlate":
		db_name = f'data/{method}/DBs/CleanPlate.db'
	else:
		db_name = f'data/{method}/DBs/cubes.db'
elif method == "cap":
	if task == "stack":
		db_name = f'data/{method}/DBs/stacking.db'
	elif task == 'pyramid':
		db_name = f'data/{method}/DBs/pyramid.db'
	elif task == 'L':
		db_name = f'data/{method}/DBs/letter_l.db'
	else:
		raise ValueError("Invalid method name")

engine = create_engine(f'sqlite:///{db_name}')
Base.metadata.create_all(engine)

Session = sessionmaker(bind=engine)
session = Session()

# Init streamlit page
st.title("Language to Control")

# init the avatars for the message icons
avatars = {"human":"images/seif_avatar.jpeg", "OD":"images/wall-e.png", "TP":"images/eve.png", "ai":"images/wall-e.png"}

def plot_trajectory(episode_id):
	try:
		ep = session.query(Episode).filter_by(id=episode_id).first()
		try:
			state_trajectory = json.loads(ep.state_trajectories)
		except:
			st.write("No state trajectory available for this episode.")
			return
		if task in ["stack", "pyramid", "L", "CleanPlate"]:
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
		else:
			state_trajectory_left = [np.array(s) for i,s in enumerate(state_trajectory) if i%2==0]
			state_trajectory_right = [np.array(s) for i,s in enumerate(state_trajectory) if i%2==1]

			efficiency_left = sum([np.linalg.norm(state_trajectory_left[i+1][:3] - state_trajectory_left[i][:3]) for i in range(len(state_trajectory_left)-1)])
			efficiency_right = sum([np.linalg.norm(state_trajectory_right[i+1][:3] - state_trajectory_right[i][:3]) for i in range(len(state_trajectory_right)-1)])

			fig = plt.figure()
			ax = fig.add_subplot(111, projection='3d')
			ax.set_title("3D trajectory. Cumulative travel = {:.2f}m + {:.2f}m".format(efficiency_left, efficiency_right))
			ax.plot([s[0] for s in state_trajectory_left], [-s[1] for s in state_trajectory_left], [s[2] for s in state_trajectory_left], label="left robot")
			ax.plot([s[0] for s in state_trajectory_right], [-s[1] for s in state_trajectory_right], [s[2] for s in state_trajectory_right], label="right robot")

			# plot start and end
			ax.scatter(state_trajectory_left[0][0], -state_trajectory_left[0][1], state_trajectory_left[0][2], c='r', marker='o')
			ax.scatter(state_trajectory_left[-1][0], -state_trajectory_left[-1][1], state_trajectory_left[-1][2], c='g', marker='o')
			ax.scatter(state_trajectory_right[0][0], -state_trajectory_right[0][1], state_trajectory_right[0][2], c='r', marker='o')
			ax.scatter(state_trajectory_right[-1][0], -state_trajectory_right[-1][1], state_trajectory_right[-1][2], c='g', marker='o')
			st.pyplot(fig)
	except Exception as e:
		return

def plot_mpc_times(episode_id):
	try:
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
	except Exception as e:
		return


def get_task_type(episode_id):
	if task in ["CleanPlate", "Sponge"]:
			return task
	
	if "ours" in method:
		instruction = session.query(Epoch).filter_by(episode_id=episode_id).order_by(Epoch.time_step.asc()).first().content
		if "stack" in instruction:
			return "stack"
		elif "L" in instruction:
			return "L"
		else:
			return "pyramid"
	else:
		return task

def append_message(message:dict):
	# Function to append a message to the chat history
	message_type = message["type"]
	if message_type == "image":
		with st.chat_message("human", avatar=avatars["human"]):
			try:
				image_path = wd + f"/data/{method}/images" + message["content"].split("images")[-1]
				st.image(image_path, width=400, caption="Current scene")
			except:
				try:
					image_path = image_path.replace(":", "_")
					st.image(image_path, width=400, caption="Current scene")
				except:
					st.markdown("Image not found")
	else:
		with st.chat_message(message_type, avatar=avatars[message_type]):
			st.markdown(message["content"])

	
# count number of episodes in db
episodes = session.query(Episode).all()  # Assuming 'id' is a primary key column
episodes = [e for e in episodes if get_task_type(e.id) == task]
episode_id = st.number_input('Enter an Episode ID', min_value=1, max_value=len(episodes), value=1)
episode_id -= 1

epochs = session.query(Epoch).filter_by(episode_id=episodes[episode_id].id).order_by(Epoch.time_step.desc()).all()
st.session_state.messages = []
for e in epochs:
	if get_task_type(e.episode_id) != task:
		continue
	if e.image != "":
		if "ours" in method:
			image_url = e.image.replace('data', f'data/{method}')
		elif method == "cap":
			image_url = e.image.replace('cap/','').replace('data', 'data/cap').replace(':','_')
		st.session_state.messages += [{'type':'image', 'content':image_url}, {"type":e.role, 'content':e.content.replace("#","\n")}]
	else:
		st.session_state.messages += [{"type":e.role, 'content':e.content.replace("#","\n")}]


if not st.session_state.messages == []:
	# read user query
	append_message(st.session_state.messages[-1])
	# Visualize last image to see if task completed succesfully
	append_message(st.session_state.messages[0])

	# plot 3D trajectory
	plot_trajectory(episodes[episode_id].id)

	# plot MPC solve times
	plot_mpc_times(episodes[episode_id].id)

	# Display chat messages from history on app rerun
	for message in st.session_state.messages:
		append_message(message)

# close db session
session.close()



	


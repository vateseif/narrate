import streamlit as st
from sqlalchemy import func
from db import Session, Epoch, Episode

# Init db session
session = Session()	

# Init streamlit page
st.title("Language to Control")

# init the avatars for the message icons
avatars = {"human":"images/seif_avatar.jpeg", "OD":"images/wall-e.png", "TP":"images/eve.png"}

def get_task_type(episode_id):
	if episode_id == 1: return "1 (stack)"
	if episode_id == 2: return "2 (L)"
	if episode_id == 3: return "3 (pyramid)"
	
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
st.session_state.messages = st.session_state.messages

session.close()

# Display chat messages from history on app rerun
for message in st.session_state.messages:
	append_message(message)

	


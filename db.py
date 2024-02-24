from config.config import DBConfig
from sqlalchemy.orm import relationship
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Integer, String, ForeignKey, JSON

Base = declarative_base()


class Episode(Base):
    __tablename__ = 'episodes'
    
    id = Column(Integer, primary_key=True)
    name = Column(String)  # Optional, if you want to name or otherwise identify episodes
    state_trajectories = Column(JSON)  # Store state trajectories as JSON
    mpc_solve_times = Column(JSON)  # Store MPC solve times as JSON
    #n_collisions = Column(Integer)
    epochs = relationship("Epoch", backref="episode")

class Epoch(Base):
    __tablename__ = 'epochs'
    
    id = Column(Integer, primary_key=True)
    episode_id = Column(Integer, ForeignKey('episodes.id'))
    time_step = Column(Integer, nullable=False)
    role = Column(String, nullable=False)
    content = Column(String, nullable=False)
    image = Column(String, nullable=False)  # Store images as binary data



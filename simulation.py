import gym
from tqdm import tqdm

from robot import Robot
from db import Episode, Epoch, Base
from core import AbstractSimulation, BASE_DIR
from config.config import SimulationConfig, RobotConfig

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker


class Simulation(AbstractSimulation):
    def __init__(self, task_name, cfg=SimulationConfig()) -> None:
        #super().__init__(cfg)

        self.cfg = cfg
        # init env
        self.env = gym.make(f"Panda{cfg.task}-v2", render=cfg.render, debug=cfg.debug)
        # init robots
        # count number of tasks solved from a plan 
        self.plan = None
        self.task_counter = 0
        self.prev_instruction = "None"

        # simulation time
        self.t = 0.
        self.robot = Robot(self.env, Session, task_name, RobotConfig(self.cfg.task))
        # count number of tasks solved from a plan 
        self.task_counter = 0
        if self.cfg.logging:
            self.session = Session()
        
    def _store_epoch_db(self, episode_id, role, content, image_url):
        session = Session()
        
        # Find the last epoch number for this episode
        last_epoch = session.query(Epoch).filter_by(episode_id=episode_id).order_by(Epoch.time_step.desc()).first()
        if last_epoch is None:
            next_time_step = 1  # This is the first epoch for the episode
        else:
            next_time_step = last_epoch.time_step + 1
        
        # Create and insert the new epoch
        epoch = Epoch(episode_id=episode_id, time_step=next_time_step, role=role, content=content, image=image_url)
        session.add(epoch)
        session.commit()
        session.close()
    
    def reset(self):
        # reset pand env
        self.observation = self.env.reset()
        # reset robot
        self.robot.reset()
        # reset controller
        self.robot.init_states(self.observation, self.t)
        # count number of tasks solved from a plan 
        self.task_counter = 0
        # init list of RGB frames if wanna save video
        if self.cfg.logging:
            if self.session is not None:
                self.session.close()
            self.session = Session()
            self.episode = Episode()  # Assuming Episode has other fields you might set
            self.session.add(self.episode)
            self.session.commit()
   
    def _start_cap(self, prompt):
        self._store_epoch_db(self.episode.id, "human", prompt, "")
        print(f"PROMPT: {prompt}")
        print(f"CONTEXT: {[el['name'] for el in self.env.objects_info]}")
        out = self.robot.lmp(prompt, self.episode, f'objects = {[el["name"] for el in self.env.objects_info]}')

if __name__=="__main__":   
    colors = ["red", "green", "blue", "orange"]
    task_names = [
        "stacking",
        "letter_l",
        "pyramid"
        ]
    prompts = [
        lambda: "make a stack of cubes on top of the {} one".format(*sample(colors, 1)),
        lambda: "rearrange cubes to write the letter L on the table. keep {} at its location".format(*sample(colors, 1)),
        lambda: "build a pyramid with the {} and {} cubes at the base and {} cube at the top. keep {} cube at its original position.".format(*(2*sample(colors, 3)))
        ]

    from tqdm import tqdm
    from random import sample
    import time
    import multiprocessing

    N_EXPERIMENTS = 10
    for prompt_f, task_name in zip(prompts[::-1], task_names[::-1]):
        print("Running experiments for task: ", task_name)

        db_name = f"data/DBs/cap/{task_name}.db"
        engine = create_engine(f'sqlite:///{db_name}')
        Base.metadata.create_all(engine)
        Session = sessionmaker(bind=engine)
        
        def run_task():
            time.sleep(5)
            s = Simulation(task_name)
            s.reset()
            time.sleep(3)

            for i in tqdm(range(N_EXPERIMENTS)):
                try:
                    s._start_cap(prompt_f())
                except Exception as e:
                    print(e)
                s.reset()
                time.sleep(2)
        
        p = multiprocessing.Process(target=run_task)
        p.start()
        p.join()
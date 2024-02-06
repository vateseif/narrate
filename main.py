import gym
import panda_gym
import asyncio
from time import sleep
from queue import Queue
from robot import BaseRobot
from config.config import BaseRobotConfig


class Simulation:
    def __init__(self) -> None:
        self.env = gym.make(f"PandaCubes-v2", render=True)
        self.robot = BaseRobot((self.env.robots_info, self.env.objects_info),BaseRobotConfig('L'))

        self.q = asyncio.Queue()

    def reset(self):
        self.t = 0.
        self.obs = self.env.reset()
        print(self.obs)

    async def ui(self, q):
        while True:
            # Run the input call in a separate thread to avoid blocking the event loop
            user_input = await asyncio.to_thread(input, "Enter something for the continuously running function: ")
            await q.put(user_input)

    async def step(self, q):
        while True:
            if not q.empty():
                message = await q.get()
                print("Calling OD...")
                self.robot.next_plan(message, self.obs)
                #print(f"Received from input function: {message}")

            self.robot.init_states(self.obs, self.t)
            action = self.robot.step()
            self.obs, _, done, _ = self.env.step(action)
            await asyncio.sleep(.05)

    async def run(self):
        task1 = asyncio.create_task(self.ui(self.q))
        task2 = asyncio.create_task(self.step(self.q))
        await asyncio.gather(task1, task2)


if __name__ == "__main__":
    s = Simulation()
    s.reset()
    asyncio.run(s.run())

    """sleep(10)
    print("Calling LLM...")
    s.robot.next_plan("move gripper 0.1m above cube_1", s.obs)"""
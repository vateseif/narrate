from simulation import Simulation

from time import sleep
from llm import Optimization

if __name__ == "__main__":

  # Run simulator
  sim = Simulation()
  sim.run()

  #sim.create_plan("Stack all cubes on top of cube_2.")
  #sim.next_task()
  
  """
  sleep(1)

  
  optimization = Optimization(
    objective = "ca.norm_2(x - cube_4)**2",
    constraints = [
      "0.04685 - ca.norm_2(x - cube_1)",
      "0.04685 - ca.norm_2(x - cube_2)",
      "0.04685 - ca.norm_2(x - cube_3)",
      "0.04685 - ca.norm_2(x - cube_4)"
    ]  
  )
  sim.robot.MPC.apply_gpt_message(optimization, sim.observation)


  sleep(20)
  
  sim.robot.close_gripper()

  sleep(1)

  optimization = Optimization(
    objective = "ca.norm_2(x - cube_3 + np.array([0., 0.0468, 0.]))**2",
    constraints = [
      "0.04685 - ca.norm_2(x - cube_1)",
      "0.04685 - ca.norm_2(x - cube_2)",
      "0.04685 - ca.norm_2(x - cube_3)"
    ]  
  )
  sim.robot.MPC.apply_gpt_message(optimization, sim.observation)
  """

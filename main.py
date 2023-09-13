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
    objective = "ca.norm_2(x - sponge)**2",
    constraints = [
      "0.04685 - ca.norm_2(x - sponge)",
      "0.04685 - ca.norm_2(x - plate)"
    ]  
  )
  sim.robot.MPC.apply_gpt_message(optimization, sim.observation)


  sleep(7)
  
  sim.robot.close_gripper()

  sleep(2)
  optimization = Optimization(
    objective = "ca.norm_2(x - plate - np.array([0., 0., 0.02]))**2",
    constraints = [
      "0.045 - ca.norm_2(x - plate + np.array([0., 0.04, 0.0]))",
      "0.045 - ca.norm_2(x - plate - np.array([0., 0.04, 0.0]))",
    ]  
  )
  sim.robot.MPC.apply_gpt_message(optimization, sim.observation)

  sleep(5)

  optimization = Optimization(
    objective = "ca.norm_2(x - plate - np.array([0.05*ca.cos(2*t), 0.05*ca.sin(2*t), 0.02]))**2",
    constraints = []  
  )
  sim.robot.MPC.apply_gpt_message(optimization, sim.observation)
  """
  

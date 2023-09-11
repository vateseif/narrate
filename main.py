# NOTE: use conda activate safepanda for this env
from simulation import Simulation




if __name__ == "__main__":

  # simulator
  sim = Simulation()

  sim.run()

  #sim.create_plan("Stack all cubes on top of cube_2.")
  #sim.next_task()
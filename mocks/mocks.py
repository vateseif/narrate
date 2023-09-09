from llm import Plan

optimization_mock_plan = Plan(tasks=[
  "move the gripper to cube_4 and avoid any collision with every cube",
  "close gripper",
  "move the gripper above cube_2 and avoid colliding with cube_2, cube_3 and cube_1",
  "open gripper",
  "move gripper to cube_3 and avoid collisions with every cube",
  "close gripper",
  "move gripper above of cube_4 and avoid collisions with every cube apart from cube_3",
  "open gripper",
  "move gripper to cube_1 and avoid collisions with every cube",
  "close gripper",
  "move gripper above cube_3 and avoid collisions with every cube apart from cube_1",
  "open gripper"
])

optimization_mock_plan = Plan(tasks=[
  "move the gripper to cube_4 and avoid any collision with every cube",
  "close gripper",
  "move the gripper next to cube_3 and avoid colliding with cube_2, cube_3 and cube_1",
  "open gripper",
  "move gripper to cube_2 and avoid collisions with every cube",
  "close gripper",
  "move gripper at a height above cube_3 and cube_4 and in between them. Also avoid collisions with every cube apart from cube_2",
  "open gripper"
])

optimization_mock_plan = Plan(tasks=[
  "move the gripper to cube_4 and avoid any collision with every cube",
  "close gripper",
  "move the gripper in front cube_2 and avoid colliding with cube_2, cube_3 and cube_1",
  "open gripper",
  "move gripper to cube_3 and avoid collisions with every cube",
  "close gripper",
  "move gripper in behind cube_2 and avoid collisions with cube_1, cube_2, cube_4",
  "open gripper",
  "move gripper to cube_1 and avoid collisions with every cube",
  "close gripper",
  "move gripper to the right of cube_3 and avoid colliding with cube_3",
  "open gripper",
])

objective_mock_plan = Plan(tasks=[
  "move the gripper above cube_4",
  "move gripper to the position of cube_4",
  "close gripper",
  "move the gripper above cube_2 and avoid collisions with every cube",
  "open gripper",
  "move gripper above cube_3 and avoid collisions with every cube",
  "move gripper to the position of cube_3",
  "close gripper",
  "move the gripper above cube_4 and avoid collisions with every cube",
  "open gripper",
  "move gripper to above cube_1 and avoid collisions with every cube",
  "move gripper to the position of cube_1",
  "close gripper",
  "move gripper above cube_3 and avoid collisions with every cube apart from cube_1",
  "open gripper"
])

nmpcMockOptions = {
  "nmpc_objective": objective_mock_plan,
  "nmpc_optimization": optimization_mock_plan,
}
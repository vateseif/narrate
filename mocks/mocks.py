
optimization_mock_plan_stack = '''
{"tasks":[
  "move the gripper to cube_4 and avoid any collision with every cube",
  "close gripper",
  "move the gripper on top of cube_2 and avoid colliding with cube_2, cube_3 and cube_1",
  "open gripper",
  "move gripper to cube_3 and avoid collisions with every cube",
  "close gripper",
  "move gripper above of cube_4 and avoid collisions with every cube apart from cube_3",
  "open gripper",
  "move gripper to cube_1 and avoid collisions with every cube",
  "close gripper",
  "move gripper above cube_3 and avoid collisions with every cube apart from cube_1",
  "open gripper"
]}
'''

optimization_mock_plan_pyramid = '''
{"tasks":[
  "move the gripper to cube_4 and avoid any collision with every cube",
  "close gripper",
  "move the gripper next to cube_3 and avoid colliding with cube_2, cube_3 and cube_1",
  "open gripper",
  "move gripper to cube_2 and avoid collisions with every cube",
  "close gripper",
  "move gripper at a height above cube_3 and cube_4 and in between them. Also avoid collisions with every cube apart from cube_2",
  "open gripper"
]}
'''

optimization_mock_plan_L = '''
{"tasks":[
  "move the gripper to cube_4 and avoid any collision with every cube",
  "close gripper",
  "move the gripper to the left side (y axis) of cube_3 and avoid colliding with cube_1, cube_2 and cube_3",
  "open gripper",
  "move gripper to cube_2 and avoid collisions with every cube",
  "close gripper",
  "move gripper behind cube_3 and avoid collisions with cube_1, cube_3, cube_4",
  "open gripper",
  "move gripper to cube_1 and avoid collisions with every cube",
  "close gripper",
  "move gripper behind cube_2 and avoid colliding with cube_1, cube_3, cube_4",
  "open gripper",
]} 
'''

objective_mock_plan = '''
{"tasks":[
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
]}
'''

optimization_mock_plan_clean_plate = '''
{"tasks":[
  "move the gripper to sponge and avoid colliding with plate and sponge",
  "close gripper",
  "move the gripper 0.05m above the plate",
  "move the x and y coordinates of the gripper around a circle of radius 0.05m centered at the plate. Keep the z coordinate at 0.015m above the plate"
]}
'''

OD_mock = '''
{
  "objective": "ca.norm_2(x - sponge)**2",
  "constraints": [
    "0.03 - ca.norm_2(x - sponge)",
    "0.05 - ca.norm_2(x - plate)"
    ]
} 
''' 

nmpcMockOptions = {
  "nmpc_objective": objective_mock_plan,
  "stack": optimization_mock_plan_stack,
  "pyramid": optimization_mock_plan_pyramid,
  "L": optimization_mock_plan_L,
  "clean_plate": optimization_mock_plan_clean_plate,
  "OD": OD_mock
}
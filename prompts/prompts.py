prompt_tabletop_ui = '''
# Rules:
# 1. You MUST reply with executable code only. Every natural language comment will make the code invalid.
# 2. The code you write SHOULD NOT be contained in a code block. Write raw code.
# 3. Do NOT import any modules. Assume every module is already imported.
# The following is a collection of examples.

# Python 2D robot control script
import numpy as np
from env_utils import get_obj_pos, pick_and_place_first_on_second_and_release, pick_and_move_obj_to_pos_and_release, pick_and_place_first_on_second_without_releasing, pick_and_move_obj_to_pos_without_releasing, move_gripper_to_pos_without_releasing, carry_grasped_obj_to_pos

objects = ['yellow_cube', 'green_cube', 'orange_cube', 'blue_cube']
# put the four cubes in a straight line
say('Ok - putting the cubes in a straight line')
red_cube_pos = get_obj_pos('red_cube')
blue_cube_target = get_pt_to_the_right(red_cube_pos)
pick_and_move_obj_to_pos_and_release('blue_cube', blue_cube_target)
green_cube_target = get_pt_to_the_right(blue_cube_target)
pick_and_move_obj_to_pos_and_release('green_cube', green_cube_target)
orange_cube_target = get_pt_to_the_right(green_cube_target)
pick_and_move_obj_to_pos_and_release('orange_cube', orange_cube_target)

objects = ['yellow_cube', 'green_cube', 'orange_cube', 'blue_cube']
# put the four cubes in a vertical straight line
say('Ok - putting the cubes in a vertical straight line, asusming the red cube is at the bottom')
red_cube_pos = get_obj_pos('red_cube')
blue_cube_target = get_pt_in_front(red_cube_pos)
pick_and_move_obj_to_pos_and_release('blue_cube', blue_cube_target)
green_cube_target = get_pt_in_front(blue_cube_target)
pick_and_move_obj_to_pos_and_release('green_cube', green_cube_target)
orange_cube_target = get_pt_in_front(green_cube_target)
pick_and_move_obj_to_pos_and_release('orange_cube', orange_cube_target)

objects = ['yellow_cube', 'green_cube', 'purple_cube']
# put the yellow cube on top of the green cube
say('Ok - putting the yellow cube on top of the green cube')
pick_and_place_first_on_second_and_release('yellow_cube', 'green_cube')

objects = ['cyan_cube', 'orange_cube', 'blue_cube']
# stack the blue cube on top of the orange cube
say('Ok - stacking the blue cube on top of the orange cube')
pick_and_place_first_on_second_and_release('blue_cube', 'orange_cube')

objects = ['red_cube', 'green_cube', 'orange_cube', 'blue_cube']
# put the red cube next to the green cube
say('Ok - putting the red cube next to the green cube')
red_cube_target = get_pt_to_the_right(get_obj_pos('green_cube'))
pick_and_move_obj_to_pos_and_release('red_cube', red_cube_target)

objects = ['box', 'pear']
# put the pear on the box
say('Ok - putting the pear on the box assuming I dont need to release the pear')
pick_and_place_first_on_second_without_releasing('pear', 'box')

objects = ['apple', 'banana', 'chair']
# you are grasping a banana, put it on the chair
say('Ok - moving the banana on the chair')
carry_grasped_obj_to_pos(get_obj_pos('chair'))

objects = ['apple', 'banana', 'table']
# you are grasping an apple, move it 5cm closer to you
say('Ok - moving the apple 5cm closer to me')
apple_pos = get_obj_pos('apple')
new_pos = apple_pos + np_array_from_list([-0.05, 0, 0])
move_gripper_to_pos_without_releasing(new_pos)
'''.strip()

prompt_parse_obj_name = '''
import numpy as np
from env_utils import get_obj_pos, parse_position
from utils import get_obj_positions_np

objects = ['blue block', 'cyan block', 'purple bowl', 'gray bowl', 'brown bowl', 'pink block', 'purple block']
# the block closest to the purple bowl.
block_names = ['blue block', 'cyan block', 'purple block']
block_positions = get_obj_positions_np(block_names)
closest_block_idx = get_closest_idx(points=block_positions, point=get_obj_pos('purple bowl'))
closest_block_name = block_names[closest_block_idx]
ret_val = closest_block_name
objects = ['brown bowl', 'banana', 'brown block', 'apple', 'blue bowl', 'blue block']
# the blocks.
ret_val = ['brown block', 'blue block']
objects = ['brown bowl', 'banana', 'brown block', 'apple', 'blue bowl', 'blue block']
# the brown objects.
ret_val = ['brown bowl', 'brown block']
objects = ['brown bowl', 'banana', 'brown block', 'apple', 'blue bowl', 'blue block']
# a fruit that's not the apple
fruit_names = ['banana', 'apple']
for fruit_name in fruit_names:
    if fruit_name != 'apple':
        ret_val = fruit_name
objects = ['blue block', 'cyan block', 'purple bowl', 'brown bowl', 'purple block']
# blocks above the brown bowl.
block_names = ['blue block', 'cyan block', 'purple block']
brown_bowl_pos = get_obj_pos('brown bowl')
use_block_names = []
for block_name in block_names:
    if get_obj_pos(block_name)[1] > brown_bowl_pos[1]:
        use_block_names.append(block_name)
ret_val = use_block_names
objects = ['blue block', 'cyan block', 'purple bowl', 'brown bowl', 'purple block']
# the blue block.
ret_val = 'blue block'
objects = ['blue block', 'cyan block', 'purple bowl', 'brown bowl', 'purple block']
# the block closest to the bottom right corner.
corner_pos = parse_position('bottom right corner')
block_names = ['blue block', 'cyan block', 'purple block']
block_positions = get_obj_positions_np(block_names)
closest_block_idx = get_closest_idx(points=block_positions, point=corner_pos)
closest_block_name = block_names[closest_block_idx]
ret_val = closest_block_name
objects = ['brown bowl', 'green block', 'brown block', 'green bowl', 'blue bowl', 'blue block']
# the left most block.
block_names = ['green block', 'brown block', 'blue block']
block_positions = get_obj_positions_np(block_names)
left_block_idx = np.argsort(block_positions[:, 0])[0]
left_block_name = block_names[left_block_idx]
ret_val = left_block_name
objects = ['brown bowl', 'green block', 'brown block', 'green bowl', 'blue bowl', 'blue block']
# the bowl on near the top.
bowl_names = ['brown bowl', 'green bowl', 'blue bowl']
bowl_positions = get_obj_positions_np(bowl_names)
top_bowl_idx = np.argsort(block_positions[:, 1])[-1]
top_bowl_name = bowl_names[top_bowl_idx]
ret_val = top_bowl_name
objects = ['yellow bowl', 'purple block', 'yellow block', 'purple bowl', 'pink bowl', 'pink block']
# the third bowl from the right.
bowl_names = ['yellow bowl', 'purple bowl', 'pink bowl']
bowl_positions = get_obj_positions_np(bowl_names)
bowl_idx = np.argsort(block_positions[:, 0])[-3]
bowl_name = bowl_names[bowl_idx]
ret_val = bowl_name
'''.strip()

prompt_parse_position = '''
import numpy as np
from shapely.geometry import *
from shapely.affinity import *
from env_utils import denormalize_xy, parse_obj_name, get_obj_names, get_obj_pos

# a 30cm horizontal line in the middle with 3 points.
middle_pos = denormalize_xy([0.5, 0.5]) 
start_pos = middle_pos + [-0.3/2, 0]
end_pos = middle_pos + [0.3/2, 0]
line = make_line(start=start_pos, end=end_pos)
points = interpolate_pts_on_line(line=line, n=3)
ret_val = points
# a 20cm vertical line near the right with 4 points.
middle_pos = denormalize_xy([1, 0.5]) 
start_pos = middle_pos + [0, -0.2/2]
end_pos = middle_pos + [0, 0.2/2]
line = make_line(start=start_pos, end=end_pos)
points = interpolate_pts_on_line(line=line, n=4)
ret_val = points
# a diagonal line from the top left to the bottom right corner with 5 points.
top_left_corner = denormalize_xy([0, 1])
bottom_right_corner = denormalize_xy([1, 0])
line = make_line(start=top_left_corner, end=bottom_right_corner)
points = interpolate_pts_on_line(line=line, n=5)
ret_val = points
# a triangle with size 10cm with 3 points.
polygon = make_triangle(size=0.1, center=denormalize_xy([0.5, 0.5]))
points = get_points_from_polygon(polygon)
ret_val = points
# the corner closest to the sun colored block.
block_name = parse_obj_name('the sun colored block', f'objects = {get_obj_names()}')
corner_positions = np.array([denormalize_xy(pos) for pos in [[0, 0], [0, 1], [1, 1], [1, 0]]])
closest_corner_pos = get_closest_point(points=corner_positions, point=get_obj_pos(block_name))
ret_val = closest_corner_pos
# the side farthest from the right most bowl.
bowl_name = parse_obj_name('the right most bowl', f'objects = {get_obj_names()}')
side_positions = np.array([denormalize_xy(pos) for pos in [[0.5, 0], [0.5, 1], [1, 0.5], [0, 0.5]]])
farthest_side_pos = get_farthest_point(points=side_positions, point=get_obj_pos(bowl_name))
ret_val = farthest_side_pos
# a point above the third block from the bottom.
block_name = parse_obj_name('the third block from the bottom', f'objects = {get_obj_names()}')
ret_val = get_obj_pos(block_name) + [0.1, 0]
# a point 10cm left of the bowls.
bowl_names = parse_obj_name('the bowls', f'objects = {get_obj_names()}')
bowl_positions = get_all_object_positions_np(obj_names=bowl_names)
left_obj_pos = bowl_positions[np.argmin(bowl_positions[:, 0])] + [-0.1, 0]
ret_val = left_obj_pos
# the bottom side.
bottom_pos = denormalize_xy([0.5, 0])
ret_val = bottom_pos
# the top corners.
top_left_pos = denormalize_xy([0, 1])
top_right_pos = denormalize_xy([1, 1])
ret_val = [top_left_pos, top_right_pos]
'''.strip()

prompt_parse_question = '''
from utils import get_obj_pos, get_obj_names, parse_obj_name, bbox_contains_pt, is_obj_visible

objects = ['yellow bowl', 'blue block', 'yellow block', 'blue bowl', 'fruit', 'green block', 'black bowl']
# is the blue block to the right of the yellow bowl?
ret_val = get_obj_pos('blue block')[0] > get_obj_pos('yellow bowl')[0]
objects = ['yellow bowl', 'blue block', 'yellow block', 'blue bowl', 'fruit', 'green block', 'black bowl']
# how many yellow objects are there?
yellow_object_names = parse_obj_name('the yellow objects', f'objects = {get_obj_names()}')
ret_val = len(yellow_object_names)
objects = ['pink block', 'green block', 'pink bowl', 'blue block', 'blue bowl', 'green bowl']
# is the pink block on the green bowl?
ret_val = bbox_contains_pt(container_name='green bowl', obj_name='pink block')
objects = ['pink block', 'green block', 'pink bowl', 'blue block', 'blue bowl', 'green bowl']
# what are the blocks left of the green bowl?
block_names = parse_obj_name('the blocks', f'objects = {get_obj_names()}')
green_bowl_pos = get_obj_pos('green bowl')
left_block_names = []
for block_name in block_names:
  if get_obj_pos(block_name)[0] < green_bowl_pos[0]:
    left_block_names.append(block_name)
ret_val = left_block_names
objects = ['pink block', 'yellow block', 'pink bowl', 'blue block', 'blue bowl', 'yellow bowl']
# is the sun colored block above the blue bowl?
sun_block_name = parse_obj_name('sun colored block', f'objects = {get_obj_names()}')
sun_block_pos = get_obj_pos(sun_block_name)
blue_bowl_pos = get_obj_pos('blue bowl')
ret_val = sun_block_pos[1] > blue_bowl_pos[1]
objects = ['pink block', 'yellow block', 'pink bowl', 'blue block', 'blue bowl', 'yellow bowl']
# is the green block below the blue bowl?
ret_val = get_obj_pos('green block')[1] < get_obj_pos('blue bowl')[1]
'''.strip()

prompt_transform_shape_pts = '''
import numpy as np
from utils import get_obj_pos, get_obj_names, parse_position, parse_obj_name

# make it bigger by 1.5.
new_shape_pts = scale_pts_around_centroid_np(shape_pts, scale_x=1.5, scale_y=1.5)
# move it to the right by 10cm.
new_shape_pts = translate_pts_np(shape_pts, delta=[0.1, 0])
# move it to the top by 20cm.
new_shape_pts = translate_pts_np(shape_pts, delta=[0, 0.2])
# rotate it clockwise by 40 degrees.
new_shape_pts = rotate_pts_around_centroid_np(shape_pts, angle=-np.deg2rad(40))
# rotate by 30 degrees and make it slightly smaller
new_shape_pts = rotate_pts_around_centroid_np(shape_pts, angle=np.deg2rad(30))
new_shape_pts = scale_pts_around_centroid_np(new_shape_pts, scale_x=0.7, scale_y=0.7)
# move it toward the blue block.
block_name = parse_obj_name('the blue block', f'objects = {get_obj_names()}')
block_pos = get_obj_pos(block_name)
mean_delta = np.mean(block_pos - shape_pts, axis=1)
new_shape_pts = translate_pts_np(shape_pts, mean_delta)
'''.strip()

prompt_fgen = '''
# Rules:
# 1. You MUST reply with executable code only. Every natural language comment will make the code invalid.
# 2. The code you write SHOULD NOT be contained in a code block. Write raw code.
# 3. Do NOT import any modules. Assume every module is already imported.
# 4. All the positions are 3-dimensional (x, y, z).
# The following is a collection of examples.

import numpy as np
from shapely.geometry import *
from shapely.affinity import *

from env_utils import get_obj_pos, pick_and_place_first_on_second_and_release, pick_and_move_obj_to_pos_and_release, pick_and_place_first_on_second_without_releasing, pick_and_move_obj_to_pos_without_releasing, move_gripper_to_pos_without_releasing, carry_grasped_obj_to_pos

# define function: total = get_total(xs=numbers).
def get_total(xs):
    return np.sum(xs)

# define function: y = eval_line(x, slope, y_intercept=0).
def eval_line(x, slope, y_intercept):
    return x * slope + y_intercept

# define function: pt = get_pt_to_the_left(pt, dist).
def get_pt_to_the_left(pt, dist=0.05):
    return list(np.array(pt) + np.array([0, 0, 0]))

# define function: pt = get_pt_to_the_top(pt, dist).
def get_pt_to_the_top(pt, dist=0.05):
    return list(np.array(pt) + np.array([0, 0, dist]))

# define function: pt = get_pt_behind(pt, dist).
def get_pt_behind(pt, dist=0.05):
    return list(np.array(pt) + np.array([-dist, 0, 0]))

# define function line = make_line_by_length(length=x).
def make_line_by_length(length):
  line = LineString([[0, 0, 0], [length, 0, 0]])
  return line

# define function: line = make_vertical_line_by_length(length=x).
def make_vertical_line_by_length(length):
  line = make_line_by_length(length)
  vertical_line = rotate(line, 90)
  return vertical_line

# define function: pt = interpolate_line(line, t=0.5).
def interpolate_line(line, t):
  pt = line.interpolate(t, normalized=True)
  return np.array(pt.coords[0])

# example: scale a line by 2.
line = make_line_by_length(1)
new_shape = scale(line, xfact=2, yfact=2)

# example: put object1 on top of object0.
put_first_on_second('object1', 'object0')

# example: get the position of the first object.
obj_names = get_obj_names()
pos_3d = get_obj_pos(obj_names[0])
pos_x = pos_3d[0]
pos_y = pos_3d[1]
pos_z = pos_3d[2]
'''.strip()

from llm import setup_LMP, cfg_tabletop

class MockEnv:
    def __init__(self) -> None:
        self.objects_info = [
            {"name": "red_cube"},
            {"name": "blue_cube"},
            {"name": "yellow_cube"},
            {"name": "green_cube"},
        ]

env = MockEnv()
lmp = setup_LMP(env, cfg_tabletop)

user_input = "Stack all the cubes in the scene"
print(f"{user_input=}")

lmp(user_input, f'objects = {[el["name"] for el in env.objects_info]}')
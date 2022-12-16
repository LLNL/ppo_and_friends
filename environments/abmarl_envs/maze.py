import os
import numpy as np
from ppo_and_friends.environments.abmarl_envs.maze_sim import AlternateMazeNavigationSim
from abmarl.examples import MazeNavigationAgent
from abmarl.sim.gridworld.agent import GridWorldAgent
from abmarl.managers import AllStepManager
from abmarl.external import MultiAgentWrapper
from abmarl.sim.wrappers import FlattenWrapper

object_registry = {
    'N': lambda n: MazeNavigationAgent(
        id='navigator',
        encoding=1,
        view_range=1,
        render_color='blue',
    ),
    'T': lambda n: GridWorldAgent(
        id='target',
        encoding=3,
        render_color='green'
    ),
    'W': lambda n: GridWorldAgent(
        id=f'wall{n}',
        encoding=2,
        blocking=True,
        render_shape='s'
    )
}

file_name = os.path.join(os.path.realpath(
    os.path.dirname(__file__)), "maze.txt")

sim = MultiAgentWrapper(
        AllStepManager(
            FlattenWrapper(
                AlternateMazeNavigationSim.build_sim_from_file(
                    file_name,
                    object_registry,
                    overlapping={1: [3], 3: [1]},
                    observe="position",
                )
            )
        )
    )

abmarl_maze_env = sim

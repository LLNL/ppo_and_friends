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

small_maze = os.path.join(os.path.realpath(
    os.path.dirname(__file__)), "maze.txt")

large_maze = os.path.join(os.path.realpath(
    os.path.dirname(__file__)), "large_maze.txt")

sm_sim = MultiAgentWrapper(
        AllStepManager(
            FlattenWrapper(
                AlternateMazeNavigationSim.build_sim_from_file(
                    small_maze,
                    object_registry,
                    overlapping={1: [3], 3: [1]},
                    observe="grid",
                )
            )
        )
    )

sm_blind_sim = MultiAgentWrapper(
        AllStepManager(
            FlattenWrapper(
                AlternateMazeNavigationSim.build_sim_from_file(
                    small_maze,
                    object_registry,
                    overlapping={1: [3], 3: [1]},
                    observe="position",
                )
            )
        )
    )

lg_sim = MultiAgentWrapper(
        AllStepManager(
            FlattenWrapper(
                AlternateMazeNavigationSim.build_sim_from_file(
                    large_maze,
                    object_registry,
                    overlapping={1: [3], 3: [1]},
                    observe="grid",
                    max_steps=2048,
                )
            )
        )
    )

lg_blind_sim = MultiAgentWrapper(
        AllStepManager(
            FlattenWrapper(
                AlternateMazeNavigationSim.build_sim_from_file(
                    large_maze,
                    object_registry,
                    overlapping={1: [3], 3: [1]},
                    observe="position",
                    max_steps=4096,
                )
            )
        )
    )

sm_abmarl_blind_maze = sm_blind_sim
sm_abmarl_maze       = sm_sim
lg_abmarl_blind_maze = lg_blind_sim
lg_abmarl_maze       = lg_sim


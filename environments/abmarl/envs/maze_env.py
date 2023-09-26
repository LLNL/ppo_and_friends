import os
import numpy as np
from abmarl.examples import MazeNavigationAgent, MazeNavigationSim
from abmarl.sim.gridworld.agent import GridWorldAgent
from abmarl.managers import AllStepManager
from abmarl.external import MultiAgentWrapper
from abmarl.sim.wrappers import FlattenWrapper


class MazeNavigationSim2(MazeNavigationSim):

    def step(self, action_dict, **kwargs):
        #
        # NOTE: This is identical to Abmarl's original sim, except that
        # we don't punish for "bad" moves. I find that punishment really
        # impedes learning.
        #

        # Process moves
        action = action_dict['navigator']
        move_result = self.move_actor.process_action(self.navigator, action, **kwargs)

        if self.get_all_done():
            self.rewards['navigator'] += 1

        # Entropy penalty
        self.rewards['navigator'] -= 0.01


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

sm_abmarl_maze = MultiAgentWrapper(
        AllStepManager(
            FlattenWrapper(
                MazeNavigationSim2.build_sim_from_file(
                    small_maze,
                    object_registry.copy(),
                    overlapping={1: set([3]), 3: set([1])},
                    states={'PositionState'},
                    observers={'PositionCenteredEncodingObserver'},
                )
            )
        )
    )

sm_abmarl_blind_maze = MultiAgentWrapper(
        AllStepManager(
            FlattenWrapper(
                MazeNavigationSim2.build_sim_from_file(
                    small_maze,
                    object_registry.copy(),
                    overlapping={1: set([3]), 3: set([1])},
                    states={'PositionState'},
                    observers={'AbsolutePositionObserver'},
                )
            )
        )
    )

sm_abmarl_grid_pos_maze = MultiAgentWrapper(
        AllStepManager(
            FlattenWrapper(
                MazeNavigationSim2.build_sim_from_file(
                    small_maze,
                    object_registry.copy(),
                    overlapping={1: set([3]), 3: set([1])},
                    states={'PositionState'},
                    observers={'AbsolutePositionObserver',
                        'PositionCenteredEncodingObserver'},
                )
            )
        )
    )

lg_abmarl_maze = MultiAgentWrapper(
        AllStepManager(
            FlattenWrapper(
                MazeNavigationSim2.build_sim_from_file(
                    large_maze,
                    object_registry.copy(),
                    overlapping={1: set([3]), 3: set([1])},
                    states={'PositionState'},
                    observers={'PositionCenteredEncodingObserver'},
                )
            )
        )
    )

lg_abmarl_blind_maze = MultiAgentWrapper(
        AllStepManager(
            FlattenWrapper(
                MazeNavigationSim2.build_sim_from_file(
                    large_maze,
                    object_registry.copy(),
                    overlapping={1: set([3]), 3: set([1])},
                    states={'PositionState'},
                    observers={'AbsolutePositionObserver'},
                )
            )
        )
    )

lg_abmarl_grid_pos_maze = MultiAgentWrapper(
        AllStepManager(
            FlattenWrapper(
                MazeNavigationSim2.build_sim_from_file(
                    large_maze,
                    object_registry.copy(),
                    overlapping={1: set([3]), 3: set([1])},
                    states={'PositionState'},
                    observers={'AbsolutePositionObserver',
                        'PositionCenteredEncodingObserver'},
                )
            )
        )
    )

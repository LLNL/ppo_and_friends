import os
import numpy as np
from gym.spaces import Discrete, Dict
from abmarl.examples import MazeNavigationAgent, MazeNavigationSim
from abmarl.sim.gridworld.agent import GridWorldAgent
from abmarl.managers import AllStepManager
from abmarl.external import MultiAgentWrapper
from abmarl.sim.wrappers import RavelDiscreteWrapper, RavelDiscreteActionWrapper, RavelDiscreteObservationWrapper
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

class MazeDiscreteActionWrapper():

    def __init__(self, env, **kw_args):
        self.env = env
        self.sim = env.sim
        self.action_space      = Dict({"navigator" : Discrete(8)})
        self.observation_space = env.observation_space

    def step(self, action):
        step_action = {}

        for agent_id in action:
            step_action[agent_id] = np.zeros(2)

            if action[agent_id] == 0:
                pass
            elif action[agent_id] == 1:
                #
                # Move up and right.
                #
                step_action[agent_id][0] = -1
                step_action[agent_id][1] = 1
            elif action[agent_id] == 2:
                #
                # Move right.
                #
                step_action[agent_id][0] = 0
                step_action[agent_id][1] = 1
            elif action[agent_id] == 3:
                #
                # Move down and right.
                #
                step_action[agent_id][0] = 1
                step_action[agent_id][1] = 1
            elif action[agent_id] == 4:
                #
                # Move down.
                #
                step_action[agent_id][0] = 1
                step_action[agent_id][1] = 0
            elif action[agent_id] == 5:
                #
                # Move down and left.
                #
                step_action[agent_id][0] = 1
                step_action[agent_id][1] = -1
            elif action[agent_id] == 6:
                #
                # Move left.
                #
                step_action[agent_id][0] = 0
                step_action[agent_id][1] = -1
            elif action[agent_id] == 7:
                #
                # Move up and left.
                #
                step_action[agent_id][0] = -1
                step_action[agent_id][1] = -1
            elif action[agent_id] == 8:
                #
                # Move up.
                #
                step_action[agent_id][0] = -1
                step_action[agent_id][1] = 0
            else:
                print(f"INVALID ACTION: {action}")

            move_range = self.sim.agents[agent_id].move_range
            step_action[agent_id] *= move_range

        return self.env.step(step_action)

    def reset(self):
        return self.env.reset()

    def render(self, *args, **kw_args):
        self.env.render(*args, **kw_args)

    def seed(self, seed):
        self.env.seed(seed)


file_name = os.path.join(os.path.realpath(os.path.dirname(__file__)), "maze.txt")

sim = \
MazeDiscreteActionWrapper(
    MultiAgentWrapper(
        AllStepManager(
            FlattenWrapper(
            #RavelDiscreteWrapper(
            #RavelDiscreteActionWrapper(
            #RavelDiscreteObservationWrapper(
                MazeNavigationSim.build_sim_from_file(
                    file_name,
                    object_registry,
                    overlapping={1: [3], 3: [1]},
                    observe="position",
                )
            #)
            )
        )
    )
)

policies = {
    'navigator': (
        None,
        sim.sim.agents['navigator'].observation_space,
        sim.sim.agents['navigator'].action_space,
        {}
    )
}

def policy_mapping_fn(agent_id):
    return 'navigator'

abmarl_maze_env = sim

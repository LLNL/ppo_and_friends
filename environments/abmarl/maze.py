import os
from abmarl.examples import MazeNavigationAgent, MazeNavigationSim
from abmarl.sim.gridworld.agent import GridWorldAgent
from abmarl.managers import AllStepManager
from abmarl.external import MultiAgentWrapper
from abmarl.external import GymWrapper
from abmarl.sim.wrappers import RavelDiscreteWrapper, RavelDiscreteActionWrapper, RavelDiscreteObservationWrapper
from abmarl.sim.wrappers import FlattenWrapper

object_registry = {
    'N': lambda n: MazeNavigationAgent(
        id='navigator',
        encoding=1,
        view_range=0,
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

observe  = "position"
file_name = os.path.join(os.path.realpath(os.path.dirname(__file__)), "maze.txt")

sim = MultiAgentWrapper(
        AllStepManager(
            #RavelDiscreteWrapper(
            #RavelDiscreteActionWrapper(
            #RavelDiscreteObservationWrapper(
            FlattenWrapper(
               MazeNavigationSim.build_sim_from_file(
                   file_name,
                   object_registry,
                   overlapping={1: [3], 3: [1]},
                   observe=observe,
               )
            )
        )
)

#FIXME: debugging
#import sys
#sim.reset()
#for _ in range(10):
#    action = sim.action_space.sample()
#    obs, reward, done, info = sim.step(action)
#    print("\naction, obs: \n{}\n{}\n\n".format(action, obs))
#sys.exit()

if observe == "grid":
    sim_name = "GridMazeNavigation"
else:
    sim_name = "PositionMazeNavigation"

from ray.tune.registry import register_env
register_env(sim_name, lambda sim_config: sim)


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

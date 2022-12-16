from gym.spaces import MultiBinary, Box, MultiDiscrete
from abc import ABC, abstractmethod
from ppo_and_friends.utils.mpi_utils import rank_print
from ppo_and_friends.environments.ppo_env_wrappers import IdentityWrapper
import numpy as np
from gym.spaces import Dict

from mpi4py import MPI
comm      = MPI.COMM_WORLD
rank      = comm.Get_rank()
num_procs = comm.Get_size()

class MultiBinaryEnvWrapper(ABC):

    def __init__(self, env, **kw_args):
        self.env = env
        self.action_space      = self._get_action_space()
        self.observation_space = env.observation_space

    @abstractmethod
    def _get_action_space(self):
        return

    @abstractmethod
    def step(self, action):
        return

    def reset(self):
        return self.env.reset()

    def render(self, *args, **kw_args):
        self.env.render(*args, **kw_args)

    def seed(self, seed):
        self.env.seed(seed)

class MultiBinaryCartPoleWrapper(MultiBinaryEnvWrapper):

    def _get_action_space(self):
        return MultiBinary(1)

    def step(self, action):
        return self.env.step(int(action.item()))


class MultiBinaryLunarLanderWrapper(MultiBinaryEnvWrapper):

    def _get_action_space(self):
        return MultiBinary(2)

    def step(self, action):

        step_action = None

        action = action.flatten()

        if action.sum() == 0:
           step_action = 0
        elif action.sum() == 2:
            step_action = 1
        elif action[0] == 1 and action[1] == 0:
            step_action = 2
        elif action[0] == 0 and action[1] == 1:
            step_action = 3

        return self.env.step(step_action)


class BoxIntActionWrapper():

    def __init__(self, space):
        """
        """
        if type(space) != Box or not np.issubdtype(space.dtype, np.integer):
            msg  = "ERROR: BoxIntActionWrapper only accepts spaces of "
            msg += f"type Box int. Received type {type(space)} {space.dtype}"
            rank_print(msg)
            comm.Abort()

        assert len(space.shape) == 1

        self.box_space = space

        self.range     = space.high - space.low
        self.offsets   = space.low

        self.multi_discrete_space = MultiDiscrete(self.range - space.low)

    def sample(self):
        """
        """
        return self.wrap_action(self.box_space.sample())

    def wrap_action(self, action):
        """
        """
        return action - self.offsets

    def unwrap_action(self, action):
        """
        """
        return action + self.offsets


class IdentityActionWrapper():

    def __init__(self, space):
        """
        """
        self.space = space

    def sample(self):
        """
        """
        return self.space.sample()

    def wrap_action(self, action):
        """
        """
        return action

    def unwrap_action(self, action):
        """
        """
        return action


class BoxIntActionEnvironment(ABC):

    def _wrap_action_space(self, action_space):
        """
        """
        self.action_wrappers = {}
        temp_action_space    = {}

        for agent_id in action_space:
            space = action_space[agent_id]

            if type(space) == Box and np.issubdtype(space, np.integer):
                self.action_wrappers[agent_id] = \
                    BoxIntActionWrapper(space)

                temp_action_space[agent_id] = \
                    self.action_wrappers[agent_id].multi_discrete_space

            else:
                self.action_wrappers[agent_id] = \
                    IdentityActionWrapper(space)

                temp_action_space[agent_id] = space

        self.action_space = temp_action_space

    def _action_wrapped_step(self, action):
        step_action = {}

        for agent_id in action:
            step_action[agent_id] = \
                self.action_wrappers[agent_id].unwrap_action(
                    action[agent_id])

        return self.env.step(step_action)

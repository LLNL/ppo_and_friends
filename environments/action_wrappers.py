from gym.spaces import MultiBinary, Box, MultiDiscrete
from abc import ABC, abstractmethod
from ppo_and_friends.utils.mpi_utils import rank_print
import numpy as np
from gym.spaces import Dict

from mpi4py import MPI
comm      = MPI.COMM_WORLD
rank      = comm.Get_rank()
num_procs = comm.Get_size()

class AlternateActionEnvWrapper(ABC):
    """
        This is primarily used for testing purposes. It's used to create
        a version of an environment that uses a different action space
        than it natively has.
    """

    def __init__(self, env, **kw_args):
        """
            Arguments:
                env    The environment to wrap.
        """
        self.env = env
        self.action_space      = self._get_action_space()
        self.observation_space = env.observation_space

    @abstractmethod
    def _get_action_space(self):
        """
            Return the new action space.
        """
        return

    @abstractmethod
    def step(self, action):
        """
            Step using the new action space.

            Arguments:
                action    The action to take.

            Returns:
                (obs, reward, done, info).
        """
        return

    def reset(self, *args, **kw_args):
        """
            Reset the environment.

            Returns:
                The resulting observation.
        """
        return self.env.reset(*args, **kw_args)

    def render(self, *args, **kw_args):
        """
            Render the environment.
        """
        self.env.render(*args, **kw_args)


class BoxIntActionWrapper():
    """
        Box int action spaces are odd. What we really want is MultiDiscrete.
        This class acts as a wrapper around Box int, converting it to
        MultiDiscrete.
    """

    def __init__(self, space):
        """
            Arguments:
                space    The space to wrap. This should of type Box int.
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
        self.dtype     = space.dtype

        self.multi_discrete_space = MultiDiscrete(self.range - space.low)

    def sample(self):
        """
            Sample the MultiDiscrete space.
        """
        return self.wrap_action(self.box_space.sample())

    def wrap_action(self, action):
        """
            Wrap a Box int action in MultiDiscrete.

            Arguments:
                action    The Box int action to convert to MultiDiscrete.

            Returns:
                An action converted from Box int to MultiDiscrete.
        """
        return (action - self.offsets).astype(self.dtype)

    def unwrap_action(self, action):
        """
            Unwrap a MultiDiscrete action to the Box int space.

            Arguments:
                action    A MultiDiscrete action to convert to Box int.

            Returns:
                An action converted from MultiDiscrete to Box int.
        """
        return (action + self.offsets).astype(self.dtype)


class IdentityActionWrapper():
    """
        A very simple action space wrapper that behaves exactly like
        the original space. The only difference is that we have wrap
        and unwrap methods that basically do nothing.
    """

    def __init__(self, space):
        """
            Arguments:
                space    The space to wrap.
        """
        self.space = space

    def sample(self):
        """
            Sample the action space.

            Returns:
                The sample.
        """
        return self.space.sample()

    def wrap_action(self, action):
        """
            Return the action without wrapping.
        """
        return action

    def unwrap_action(self, action):
        """
            Return the action without unwrapping.
        """
        return action


class BoxIntActionEnvironment(ABC):
    """
        An abstract environment wrapper that helps convert Box int action
        spaces to MultiDiscrete.
    """

    def _wrap_action_space(self, action_space):
        """
            Create a version of action_space where all instances of Box int
            have been converted to MultiDiscrete. Note that action_space
            is assumed to be a dictionary.

            Arguments:
                action    A dictionary mapping agent ids to action spaces.

            Returns:
                A replica of action_space where Box int spaces are converted
                to MultiDiscrete.
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

        self.action_space = Dict(temp_action_space)

    def _action_wrapped_step(self, action):
        """
            Take an action wrapped step in the environment.

            Arguments:
                action    A dictionary mapping agent ids to actions. There
                          should be no instances of Box int.

            Returns:
                The results of env.step(...)
        """
        step_action = {}

        for agent_id in action:
            step_action[agent_id] = \
                self.action_wrappers[agent_id].unwrap_action(
                    action[agent_id])

        return self.env.step(step_action)

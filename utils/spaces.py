import gym as old_gym
import gymnasium as gym
from gymnasium.spaces import Tuple, Box, Discrete, MultiDiscrete, MultiBinary
from ppo_and_friends.utils.mpi_utils import rank_print
import numpy as np

from mpi4py import MPI
comm      = MPI.COMM_WORLD
rank      = comm.Get_rank()
num_procs = comm.Get_size()

def gym_space_to_gymnasium_space(space):
    """
    gym and gymnasium spaces are incompatible. This function
    just converts gym spaces to gymnasium spaces to bypass
    the errors that crop up.

    Parameters:
    -----------
    space: gym space
        The gym space to convert.

    Returns:
    --------
    The input space converted to gymnasium.
    """
    if issubclass(type(space), old_gym.spaces.Box):
        space = gym.spaces.Box(
            low   = space.low,
            high  = space.high,
            shape = space.shape,
            dtype = space.dtype)

    elif issubclass(type(space), old_gym.spaces.Discrete):
        try:
            space = gym.spaces.Discrete(
                n     = space.n,
                start = space.start)
        except:
            space = gym.spaces.Discrete(
                n = space.n)

    elif issubclass(type(space), old_gym.spaces.MultiBinary):
        space = gym.spaces.MultiBinary(
            n = space.n)

    elif issubclass(type(space), old_gym.spaces.MultiDiscrete):
        space = gym.spaces.MultiDiscrete(
            nvec  = space.nvec,
            dtype = space.dtype)

    elif issubclass(type(space), old_gym.spaces.Dict):
        new_space = gym.spaces.Dict()

        for key in space:
            new_space[key] = gym_space_to_gymnasium_space(space[key])

        space = new_space

    elif issubclass(type(space), old_gym.spaces.Tuple):
        new_space = []

        for subspace in space:
            new_space.append(gym_space_to_gymnasium_space(subspace))

        space = gym.spaces.Tuple(new_space)

    elif ((hasattr(old_gym.spaces, "Text") and issubclass(type(space), old_gym.spaces.Text)) or
        (hasattr(old_gym.spaces, "Sequence") and issubclass(type(space), old_gym.spaces.Sequence)) or
        (hasattr(old_gym.spaces, "Graph") and issubclass(type(space), old_gym.spaces.Graph))):
        msg  = f"ERROR: conversion of gym space {space} to "
        msg += f"gymnasium is not currently supported. Contact "
        msg += f"a developer to extend support for this space."
        rank_print(msg)
        comm.abort()

    return space

class FlatteningTuple(Tuple):
    """
    A wrapper around a gymnasium Tuple space that allows us
    to get combined/flattened samples.
    """

    def __init__(self, sub_spaces, *args, **kw_args):
        """
        Parameters:
        -----------
        sub_spaces: iterable
            An iterable containing the sub-spaces to encapsulate.
        """
        super().__init__(sub_spaces, *args, **kw_args)

        self.sample_sizes   = []
        accepted_sub_spaces = [Box, Discrete, MultiDiscrete, MultiBinary]

        old_gym_spaces = [\
            old_gym.spaces.Box,
            old_gym.spaces.Discrete,
            old_gym.spaces.MultiDiscrete,
            old_gym.spaces.MultiBinary]

        for i in range(len(sub_spaces)):
            space = sub_spaces[i]

            if type(space) in old_gym_spaces:
                space = gym_space_to_gymnasium_space(space)
                sub_spaces[i] = space

            if type(space) not in accepted_sub_spaces:
                msg  = f"ERROR: sub space {space} is not currently supported by "
                msg += f"the FlatteningTuple. Supported sub-spaces are "
                msg += f"{accepted_sub_spaces}."
                rank_print(msg)
                comm.Abort()

            #
            # TODO: we could probably support multi-dimensional sub-spaces when
            # space.is_np_flattenable evaluates to True.
            #
            if len(space.shape) > 1:
                msg  = "ERROR: FlatteningTuple does not currently support "
                msg += "sub-spaces with shapes greater than 1. Given space: "
                msg += "{space}."
                rank_print(msg)
                comm.Abort()

            sample = space.sample()

            if type(sample) == np.ndarray:
                self.sample_sizes.append(sample.size)
            else:
                self.sample_sizes.append(1)

        self.sample_sizes   = np.array(self.sample_sizes, dtype=np.int32)
        self.flattened_size = self.sample_sizes.sum()

    def sample(self):
        """
        Sample the space.
        """
        return self.flatten_sample(super().sample())

    def one_hot_sample(self):
        """
        Sample the space and convert any one-hot applicable sub-spaces
        to one-hot vectors.
        """
        one_hot_sample = []

        for sub_space in self.spaces:

            if issubclass(type(sub_space), Discrete):
                sub_sample = np.zeros(sub_space.n)
                sub_sample[sub_space.sample()] = 1.0
                one_hot_sample.append(sub_sample)

            elif issubclass(type(sub_space), MultiDiscrete):
                sub_sample = np.zeros(sub_space.nvec.sum())
                sample        = sub_space.sample()

                start = 0
                for i, size in enumerate(sub_space.nvec):
                    stop = start + size
                    sub_sample[start : stop][sample[i]] = 1.0
                    start = stop

                one_hot_sample.append(sub_sample)

            else:
                one_hot_sample.append(sub_space.sample())

        return np.concatenate(one_hot_sample)

    def flatten_sample(self, sample):
        """
        Wrap a our sampled

        Parameters:
        -----------
        sample: np.ndarray
            A sample from our tuple space.

        Returns:
        --------
        A flattened version of the sample as an np.ndarray.
        """
        flattened_sample = np.zeros(self.flattened_size)

        start = 0
        for idx, sub_sample in enumerate(sample):
            stop = start + self.sample_sizes[idx]

            flattend_sample[start : stop] = sub_sample

            start = stop

        return flattened_sample

    @property
    def shape(self):
        return (self.flattened_size,)

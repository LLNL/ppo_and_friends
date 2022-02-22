import numpy as np
import torch
from .stats import RunningMeanStd
from gym.spaces import Box, Discrete, MultiDiscrete, MultiBinary
import os
import sys
import pickle
from ppo_and_friends.utils.mpi_utils import rank_print
from mpi4py import MPI

comm      = MPI.COMM_WORLD
rank      = comm.Get_rank()
num_procs = comm.Get_size()

def get_action_dtype(env):
    """
        Get our action space data type. Options are continuous
        and discrete. Note that "discrete" in this case is NOT
        the same as a "Discrete" gym space. A Box space can
        have a discrete data type.

        Arguments:
            env    The environment to query.

        Returns:
            A string representing the action space dtype.
    """
    if np.issubdtype(env.action_space.dtype, np.floating):
        return "continuous"
    elif np.issubdtype(env.action_space.dtype, np.integer):
        return "discrete"
    return "unknown"


def need_action_squeeze(env):

    need_action_squeze = False
    act_type = type(env.action_space)

    if (issubclass(act_type, Box) or
        issubclass(act_type, MultiBinary) or
        issubclass(act_type, MultiDiscrete)):

        action = env.action_space.sample()

        try:
            padded_action = np.expand_dims(action, axis=0)
            env.reset()
            env.step(padded_action)
            env.reset()
            need_action_squeeze = False
        except:
            env.reset()
            env.step(action)
            env.reset()
            need_action_squeeze = True

    elif issubclass(act_type, Discrete):
        need_action_squeeze = True
    else:
        msg  = "ERROR: unsupported action space "
        msg += "{}".format(env.action_space)
        rank_print(msg)
        comm.Abort()

    return need_action_squeeze


def update_optimizer_lr(optim, lr):
    for group in optim.param_groups:
        group['lr'] = lr


class RunningStatNormalizer(object):
    """
        A structure that allows for normalizing and de-normalizing
        data based on running stats.
    """

    def __init__(self,
                 name,
                 device,
                 epsilon = 1e-8):
        """
            Arguments:
                name        The name of the structure (used for saving/loading).
                epsilon     A very small number to help avoid 0 errors.
        """
        self.name          = name
        self.running_stats = RunningMeanStd()
        self.epsilon       = torch.tensor([epsilon]).to(device)

    def normalize(self,
                  data,
                  update_stats = True):
        """
            Normalize incoming data and potential update our stats.

            Arguments:
                data           The data to normalize.
                update_stats   Whether or not to update our runnign stats.

            Returns:
                The normalized data.
        """
        if update_stats:
            self.running_stats.update(data.cpu().numpy())

        mean     = torch.tensor(self.running_stats.mean)
        variance = torch.tensor(self.running_stats.variance)

        data = (data - mean) / torch.sqrt(variance + self.epsilon)

        return data

    def denormalize(self,
                    data):
        """
            Denormalize incoming data.

            Arguments:
                data    The data to denormalize.

            Returns:
                The denormalized data.
        """
        mean     = torch.tensor(self.running_stats.mean)
        variance = torch.tensor(self.running_stats.variance)
        data     = mean + (data * torch.sqrt(variance + self.epsilon))

        return data

    def save_info(self, path):
        """
            Save out our running stats, and check if our wrapped
            environment needs to perform any more info saves.

            Arguments:
                path    The path to save to.
        """
        f_name   = "{}_stats_{}.pkl".format(self.name, rank)
        out_file = os.path.join(path, f_name)

        with open(out_file, "wb") as fh:
            pickle.dump(self.running_stats, fh)

    def load_info(self, path):
        """
            Load our running stats and check to see if our wrapped
            environment needs to load anything.

            Arguments:
                path    The path to load from.
        """
        f_name  = "{}_stats_{}.pkl".format(self.name, rank)
        in_file = os.path.join(path, f_name)

        with open(in_file, "rb") as fh:
            self.running_stats = pickle.load(fh)

import numpy as np
import torch
from .stats import RunningMeanStd
from gymnasium.spaces import Box, Discrete, MultiDiscrete, MultiBinary, Tuple
import os
import sys
import pickle
from ppo_and_friends.utils.mpi_utils import rank_print
from mpi4py import MPI

comm      = MPI.COMM_WORLD
rank      = comm.Get_rank()
num_procs = comm.Get_size()

def get_action_dtype(action_space):
    """
        Get our action space data type.
        Arguments:
            env    The action space to query.

        Returns:
            A string representing the action space dtype.
    """
    if (type(action_space) == Box and
        np.issubdtype(action_space.dtype, np.integer)):

        msg  = "ERROR: action spaces of type Box int are not "
        msg += "directly supported. Please wrap your action space "
        msg += "in a MultiDiscrete wrapper. See "
        msg += "environments/action_wrappers for support."
        rank_print(msg)
        comm.Abort()

    if np.issubdtype(action_space.dtype, np.floating):
        return "continuous"
    elif np.issubdtype(action_space.dtype, np.integer):
        if type(action_space) == Discrete:
            return "discrete"
        elif type(action_space) == MultiBinary:
            return "multi-binary"
        elif type(action_space) == MultiDiscrete:
            return "multi-discrete"
    return "unknown"


def update_optimizer_lr(optim, lr):
    """
        Update the learning rate of an optimizer.

        Arguments:
            optim    The optimizer to update.
            lr       The new learning rate.
    """
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
                 test_mode = False,
                 epsilon   = 1e-8):
        """
            Arguments:
                name        The name of the structure (used for saving/loading).
                device      The device where data should be transfered to.
                epsilon     A very small number to help avoid 0 errors.
        """
        self.device        = device
        self.name          = name
        self.test_mode     = test_mode
        self.running_stats = RunningMeanStd()
        self.epsilon       = torch.tensor([epsilon]).to(device)

    def normalize(self,
                  data,
                  update_stats = True,
                  gather_stats = True):
        """
            Normalize incoming data and potential update our stats.

            Arguments:
                data           The data to normalize.
                update_stats   Whether or not to update our runnign stats.
                gather_stats   If update_stats is True, we can need to
                               decide whether or not to gather data across
                               processors before computing our stats.

            Returns:
                The normalized data.
        """
        if update_stats:
            self.running_stats.update(
                data.detach().cpu().numpy(),
                gather_stats)

        mean     = torch.tensor(self.running_stats.mean).to(self.device)
        variance = torch.tensor(self.running_stats.variance).to(self.device)

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
        if self.test_mode:
            return

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
        if self.test_mode:
            f_name  = "{}_stats_0.pkl".format(self.name)
        else:
            f_name  = "{}_stats_{}.pkl".format(self.name, rank)

        in_file = os.path.join(path, f_name)

        #
        # There are cases where we initially train using X ranks, and we
        # later want to continue training using (X+k) ranks. In these cases,
        # let's copy rank 0's info to all ranks > X.
        #
        if not os.path.exists(in_file):
            f_name  = "{}_stats_0.pkl".format(self.name)
            in_file = os.path.join(path, f_name)

        with open(in_file, "rb") as fh:
            self.running_stats = pickle.load(fh)

def format_seconds(seconds):
    """
        Format a floating point representing seconds into
        a nice readable string.

        Arguments:
            seconds    The seconds to format.

        Returns:
            A formatted string as either seconds, minutes, or hours.
    """

    output_time = seconds
    output_unit = "seconds"

    if output_time >= 60.:
        output_time /= 60.
        output_unit = "minutes"

        if output_time >= 60.:
            output_time /= 60.
            output_unit = "hours"

    return "{:.2f} {}".format(output_time, output_unit)

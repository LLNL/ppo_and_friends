"""
    A home for PPO specific parent networks.
"""
import torch.nn as nn
import torch
import sys
from .distributions import *
import os
from ppo_and_friends.utils.mpi_utils import rank_print
import torch.nn.functional as t_functional
from mpi4py import MPI

comm      = MPI.COMM_WORLD
rank      = comm.Get_rank()
num_procs = comm.Get_size()

class PPONetwork(nn.Module):
    """
        A base class for PPO networks.
    """

    def __init__(self,
                 name,
                 test_mode = False,
                 **kw_args):
        """
            Arguments:
                name    The name of the network.
        """

        super(PPONetwork, self).__init__()
        self.name = name
        self.test_mode = test_mode

    def save(self, path):

        if self.test_mode:
            return

        f_name = "{}_{}.model".format(self.name, rank)
        out_f  = os.path.join(path, f_name)
        torch.save(self.state_dict(), out_f)

    def load(self, path):

        if self.test_mode:
            f_name = "{}_0.model".format(self.name)
        else:
            f_name = "{}_{}.model".format(self.name, rank)

        in_f = os.path.join(path, f_name)

        #
        # There are cases where we initially train using X ranks, and we
        # later want to continue training using (X+k) ranks. In these cases,
        # let's copy rank 0's network to all ranks > X.
        #
        if not os.path.exists(in_f):
            f_name = "{}_0.model".format(self.name)
            in_f   = os.path.join(path, f_name)

        self.load_state_dict(torch.load(in_f))


class PPOActorCriticNetwork(PPONetwork):

    def __init__(self,
                 action_dtype,
                 out_dim,
                 action_nvec = None,
                 **kw_args):
        """
            NOTE: if this class is to behave as a PPO actor, it should be
            given the name "actor".

            Arguments:
                action_dtype     The data type of our action space.
                out_dim          The output dimensions.
                action_nvec      The nvec attribute of the action space.
        """

        super(PPOActorCriticNetwork, self).__init__(**kw_args)

        if action_dtype not in ["discrete", "continuous",
            "multi-binary", "multi-discrete"]:

            msg = "ERROR: unknown action type {}".format(action_dtype)
            rank_print(msg)
            comm.Abort()

        self.action_dtype = action_dtype
        self.output_func  = lambda x : x

        #
        # Actors have special roles.
        #
        if "actor" in self.name:

            if action_dtype == "discrete":
                self.distribution = CategoricalDistribution(**kw_args)
                self.output_func  = lambda x : t_functional.softmax(x, dim=-1)

            if action_dtype == "multi-discrete":
                self.distribution = MultiCategoricalDistribution(
                    nvec = action_nvec, **kw_args)
                self.output_func  = lambda x : t_functional.softmax(x, dim=-1)

            elif action_dtype == "continuous":
                self.distribution = GaussianDistribution(out_dim, **kw_args)

            elif action_dtype == "multi-binary":
                self.distribution = BernoulliDistribution(**kw_args)
                self.output_func  = t_functional.sigmoid

    def get_result(self,
                   obs,
                   testing = True):
        """
            Given an observation, return the results of performing
            inference + any other alterations that should be made
            to the result before it's fed back into the world.

            Arguments:
                obs      The observation to infer from.
                testing  Are we testing a trained environment?

            Returns:
                The predicted result with any required alterations.
        """
        if self.name == "actor":
            res = self.__call__(obs)
            res = self.distribution.refine_sample(res, testing)
            return res
        else:
            return self.__call__(obs)


class PPOConv2dNetwork(PPOActorCriticNetwork):

    def __init__(self, **kw_args):
        super(PPOConv2dNetwork, self).__init__(**kw_args)


class PPOLSTMNetwork(PPOActorCriticNetwork):

    def __init__(self, **kw_args):
        super(PPOLSTMNetwork, self).__init__(**kw_args)

    def get_zero_hidden_state(self,
                              batch_size,
                              device):
        """
            Get a hidden state tuple containing the lstm hidden state
            and cell state as zero tensors.

            Arguments:
                batch_size    The batch size to replicate.
                device        The device to send the states to.

            Returns:
                A hidden state tuple containing zero tensors.
        """

        hidden = torch.zeros(
            self.num_lstm_layers, batch_size, self.lstm_hidden_size)
        cell   = torch.zeros(
            self.num_lstm_layers, batch_size, self.lstm_hidden_size)

        hidden = hidden.to(device)
        cell   = hidden.to(device)

        return (hidden, cell)

    def reset_hidden_state(self, batch_size, device):
        """
            Reset our hidden state to zero tensors.

            Arguments:
                batch_size    The batch size to replicate.
                device        The device to send the states to.
        """
        self.hidden_state = self.get_zero_hidden_state(
            batch_size, device)


class SingleSplitObservationNetwork(PPOActorCriticNetwork):
    """
        The idea here is to support splitting the observations into
        two sub-networks before merging them back together. This is
        usually used when wanting to split proprioceptive and
        exteroceptive information.
    """

    def __init__(self,
                 split_start,
                 **kw_args):
        """
            Arguments:
                split_start      Where in the observation space the split
                                 should start.
        """

        super(SingleSplitObservationNetwork, self).__init__(**kw_args)

        if split_start <= 0:
            msg  = "ERROR: SingleSplitObservationNetwork requires a split "
            msg += "start > 0."
            rank_print(msg)
            comm.Abort()

        self.split_start = split_start

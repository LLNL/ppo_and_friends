"""
    A home for PPO specific parent networks.
"""
import torch.nn as nn
import torch
import os
from abc import ABC, abstractmethod
from ppo_and_friends.utils.mpi_utils import rank_print
from ppo_and_friends.utils.misc import get_size_and_shape

from mpi4py import MPI
comm      = MPI.COMM_WORLD
rank      = comm.Get_rank()
num_procs = comm.Get_size()


class PPONetwork(ABC, nn.Module):
    """
    A base class for PPO networks.
    """

    def __init__(self,
                 in_shape,
                 out_shape,
                 name      = "ppo-netork",
                 test_mode = False,
                 **kw_args):
        """
        Parameters:
        -----------
        in_shape: int or tuple
            The shape of the input.
        out_shape: int or tuple
            The shape of the output.
        name: str
            The name of the network.
        test_mode: bool
            Are we testing a trained policy?
        """
        super(PPONetwork, self).__init__()

        if type(in_shape) != type(None):
            self.in_size, self.in_shape = get_size_and_shape(in_shape)
        else:
            self.in_size  = None
            self.in_shape = None

        if type(out_shape) != type(None):
            self.out_size, self.out_shape = get_size_and_shape(out_shape)
        else:
            self.out_size  = None
            self.out_shape = None

        self.output_func = lambda x : x
        self.name        = name
        self.test_mode   = test_mode

    @abstractmethod
    def forward(self, *args, **kw_args):
        pass

    def _shape_output(self, output):
        """
        Reshape the network output to match our expected output shape.

        Parameters:
        -----------
        output: tensor or numpy.ndarray
            The network output

        Returns:
        --------
        tensor or numpy.ndarray:
            The output reshaped.
        """
        out_shape = (output.shape[0],) + self.out_shape
        output    = output.reshape(out_shape)
        return output

    def save(self, path):
        """
        Save our state dict to a specified path using the
        class name as an identifier.

        Parameters:
        -----------
        path: str
            The path to save to.
        """

        if self.test_mode:
            return

        f_name = "{}_{}.model".format(self.name, rank)
        out_f  = os.path.join(path, f_name)
        torch.save(self.state_dict(), out_f)

    def load(self, path):
        """
        Load a state dict that was previously save using this class.
        It's assumed that the name will match this class's name.

        Parameters:
        -----------
        path: str
            The path to save to.
        """

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

# TODO: this class is out-dated and not really needed anymore.
class PPOConv2dNetwork(PPONetwork):

    def __init__(self, *args, **kw_args):
        """
        """
        super(PPOConv2dNetwork, self).__init__(*args, **kw_args)


class PPOLSTMNetwork(PPONetwork):

    def __init__(self, *args, **kw_args):
        """
        """
        super(PPOLSTMNetwork, self).__init__(*args, **kw_args)

    def get_zero_hidden_state(self,
                              batch_size,
                              device):
        """
        Get a hidden state tuple containing the lstm hidden state
        and cell state as zero tensors.

        Parameters:
        -----------
        batch_size: int
            The batch size to replicate.
        device: torch.device
            The device to send the states to.

        Returns:
        --------
        tuple:
            A hidden state tuple containing zero tensors.
        """

        hidden = torch.zeros(
            self.num_lstm_layers, batch_size, self.lstm_hidden_size).to(torch.float32)
        cell   = torch.zeros(
            self.num_lstm_layers, batch_size, self.lstm_hidden_size).to(torch.float32)

        hidden = hidden.to(device)
        cell   = hidden.to(device)

        return (hidden, cell)

    def reset_hidden_state(self, batch_size, device):
        """
        Reset our hidden state to zero tensors.

        Parameters:
        -----------
        batch_size: int
            The batch size to replicate.
        device: torch.device
            The device to send the states to.
        """
        self.hidden_state = self.get_zero_hidden_state(
            batch_size, device)


# FIXME: this has not been tested in a long while...
class SingleSplitObservationNetwork(PPONetwork):
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
        Parameters:
        -----------
        split_start: int
            Where in the observation space the split should start.
        """

        super(SingleSplitObservationNetwork, self).__init__(**kw_args)

        if split_start <= 0:
            msg  = "ERROR: SingleSplitObservationNetwork requires a split "
            msg += "start > 0."
            rank_print(msg)
            comm.Abort()

        self.split_start = split_start

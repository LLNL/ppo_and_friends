import torch
from torch import nn
from ppo_and_friends.networks.ppo_networks.feed_forward import FeedForwardNetwork
from ppo_and_friends.networks.utils import *
from ppo_and_friends.utils.mpi_utils import rank_print
from ppo_and_friends.networks.ppo_networks.base import PPOLSTMNetwork

from mpi4py import MPI
comm      = MPI.COMM_WORLD
rank      = comm.Get_rank()
num_procs = comm.Get_size()


class LSTMNetwork(PPOLSTMNetwork):

    def __init__(self,
                 in_shape,
                 out_shape,
                 sequence_length   = 10,
                 out_init          = None,
                 activation        = nn.ReLU(),
                 lstm_hidden_size  = 128,
                 num_lstm_layers   = 1,
                 ff_hidden_size    = 128,
                 ff_hidden_depth   = 1,
                 **kw_args):
        """
        A class defining an LSTM centered network.

        NOTES: I've found that feed forward networks generally learn
        a lot faster than lstm in all of the environments I've tested
        so far. However, there are certainly situations where using an
        lstm network is advantageous. For instance, if you mask the
        velocities from even a simple environment like CartPole, the
        feed forward networks begin to lose the ability to learn much
        (which makes sense), but an LSTM network can perform just fine
        here. Some other things I've found:
            1. LSTM networks seem to be very sensitive to hyper-parameter
               tuning (much more so than MLPs).
            2. The sequence length can have a dramatic impact on
               performance. As an example, I adjusted the sequence length
               from 8 to 12 while testing on the LunarLander environment
               with masked velocities, and that adjustment made the
               difference between not learning much of anything and
               learning a robust policy in fairly few iterations.

        Parameters:
        -----------
        in_shape: tuple
            The input shape.
        out_shape: tuple
            The the output shape.
        sequence_length: int
            The length of the input sequence.
        out_init: float
            A std weight to apply to the output layer.
        activation: torch.nn function
            The activation function to use on the
            output of hidden layers.
        lstm_hidden_size: int
            The hidden size for the lstm layers.
        num_lstm_layers: int
            The the number of lstm layers to stack.
        ff_hidden_size: int
            You can optionally add hidden layers to
            the output feed forward section, and this
            determines their size.
        ff_hidden_depth: int
            You can optionally add hidden layers to
            the output feed forward section, and this
            determines its depth.
        """

        super(LSTMNetwork, self).__init__(
            in_shape  = in_shape,
            out_shape = out_shape,
            **kw_args)

        self.sequence_length  = sequence_length
        self.activation       = activation
        self.lstm_hidden_size = lstm_hidden_size
        self.num_lstm_layers  = num_lstm_layers

        self.lstm = nn.LSTM(self.in_size, lstm_hidden_size, num_lstm_layers)
        self.lstm = init_net_parameters(self.lstm)
        self.layer_norm = nn.LayerNorm(lstm_hidden_size)

        self.hidden_state = None

        ff_kw_args = kw_args.copy()
        ff_kw_args["name"] = self.name + "_lstm_ff"

        self.ff_layers = FeedForwardNetwork(
            in_shape     = lstm_hidden_size,
            out_shape    = self.out_shape,
            hidden_size  = ff_hidden_size,
            hidden_depth = ff_hidden_depth,
            activation   = self.activation,
            is_embedded  = False,
            out_init     = out_init,
            **ff_kw_args)

    def forward(self, _input):

        if len(_input.shape) == 2:
            out = _input.unsqueeze(0)
        else:
            out = torch.transpose(_input, 0, 1)

        batch_size = out.shape[1]

        if (self.hidden_state == None or
            self.hidden_state[0].shape[1] != batch_size):
            self.reset_hidden_state(batch_size, out.device)

        _, self.hidden_state = self.lstm(out, self.hidden_state)

        out = self.hidden_state[0][-1]

        out = self.layer_norm(out)
        out = self.activation(out)

        out = self.ff_layers(out)

        out = self.output_func(out)

        return self._shape_output(out)

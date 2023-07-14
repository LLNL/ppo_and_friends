import torch
from torch import nn
from functools import reduce
from ppo_and_friends.networks.utils import *
from ppo_and_friends.utils.mpi_utils import rank_print
from ppo_and_friends.networks.ppo_networks import PPOLSTMNetwork

from mpi4py import MPI
comm      = MPI.COMM_WORLD
rank      = comm.Get_rank()
num_procs = comm.Get_size()


class LSTMNetwork(PPOLSTMNetwork):

    def __init__(self,
                 in_dim,
                 out_dim,
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

            Arguments:
                in_dim            The dimensions of the input data. For
                                  instance, if the expected input shape is
                                  (length, batch_size, 16), in_dim would be
                                  (length, 16).
                out_dim           The expected dimensions for the output. For
                                  instance, if the expected output shape is
                                  (length, batch_size, 16), out_dim would be
                                  (length, 16,).
                sequence_length   The length of the input sequence.
                out_init          A std weight to apply to the output layer.
                activation        The activation function to use on the
                                  output of hidden layers.
                lstm_hidden_size  The hidden size for the lstm layers.
                num_lstm_layers   The the number of lstm layers to stack.
                ff_hidden_size    You can optionally add hidden layers to
                                  the output feed forward section, and this
                                  determines their size.
                ff_hidden_depth   You can optionally add hidden layers to
                                  the output feed forward section, and this
                                  determines its depth.
        """

        super(LSTMNetwork, self).__init__(
            out_dim = out_dim,
            **kw_args)

        if type(out_dim) == tuple:
            out_size     = reduce(lambda a, b: a*b, out_dim)
            self.out_dim = out_dim
        else:
            out_size     = out_dim
            self.out_dim = (out_dim,)

        self.sequence_length  = sequence_length
        self.activation       = activation
        self.lstm_hidden_size = lstm_hidden_size
        self.num_lstm_layers  = num_lstm_layers

        self.lstm = nn.LSTM(in_dim, lstm_hidden_size, num_lstm_layers)
        self.lstm = init_net_parameters(self.lstm)
        self.layer_norm = nn.LayerNorm(lstm_hidden_size)

        self.hidden_state = None

        ff_kw_args = kw_args.copy()
        ff_kw_args["name"] = self.name + "_lstm_ff"

        self.ff_layers = FeedForwardNetwork(
            in_dim       = lstm_hidden_size,
            hidden_size  = ff_hidden_size,
            hidden_depth = ff_hidden_depth,
            out_dim      = out_size,
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

        return out

import torch
import os
from torch import nn
import torch.nn.functional as F
from functools import reduce
import numpy as np
import sys
from ppo_and_friends.networks.utils import *
from ppo_and_friends.utils.mpi_utils import rank_print
from .ppo_networks import PPOActorCriticNetwork, PPOConv2dNetwork, SingleSplitObservationNetwork
from .ppo_networks import PPOLSTMNetwork

from mpi4py import MPI
comm      = MPI.COMM_WORLD
rank      = comm.Get_rank()
num_procs = comm.Get_size()


class FeedForwardNetwork(PPOActorCriticNetwork):

    def __init__(self,
                 in_dim,
                 out_dim,
                 out_init     = None,
                 activation   = nn.ReLU(),
                 hidden_size  = 128,
                 hidden_depth = 3,
                 is_embedded  = False,
                 **kw_args):
        """
            A class defining a customizable feed-forward network.

            Arguments:
                in_dim          The dimensions of the input data. For
                                instance, if the expected input shape is
                                (batch_size, 16, 4), in_dim would be (16, 4).
                out_dim         The expected dimensions for the output. For
                                instance, if the expected output shape is
                                (batch_size, 16, 4), out_dim would be (16, 4).
                out_init        A std weight to apply to the output layer.
                activation      The activation function to use on the output
                                of hidden layers.
                hidden_size     Can either be an int or list of ints. If an int,
                                all layers will be this size. Otherwise, a list
                                designates the size for each layer. Note that
                                the hidden_depth argument is ignored if this
                                argument is a list and the depth is instead
                                taken from the length of the list. Note that
                                this argument can be set to 0 or an empty list,
                                resulting in only an input and output layer.
                hidden_depth    The number of hidden layers. Note that this is
                                ignored if hidden_size is a list.
                is_embedded     If True, this network will be treated as being
                                embedded in a large network, and the output
                                from the output layer will be sent through
                                the activation function.
        """

        super(FeedForwardNetwork, self).__init__(
            out_dim = out_dim,
            **kw_args)

        self.is_embedded   = is_embedded
        self.have_hidden   = True
        self.two_layer_net = False

        if type(out_dim) == tuple:
            out_size     = reduce(lambda a, b: a*b, out_dim)
            self.out_dim = out_dim
        else:
            out_size     = out_dim
            self.out_dim = (out_dim,)

        self.activation = activation

        self.sequential_net = \
            create_sequential_network(
                in_dim       = in_dim,
                out_size     = out_size,
                hidden_size  = hidden_size,
                hidden_depth = hidden_depth,
                activation   = activation,
                out_init     = out_init)

    def forward(self, _input):
        out = _input.flatten(start_dim = 1)
        out = self.sequential_net(out)

        #
        # If this network is embedded in a larger network,
        # we want to treat it as such and return the output
        # after activation.
        #
        if self.is_embedded:
            return self.activation(out)

        out = self.output_func(out)

        out_shape = (out.shape[0],) + self.out_dim
        out = out.reshape(out_shape)

        return out


class SplitObsNetwork(SingleSplitObservationNetwork):
    def __init__(self,
                 in_dim,
                 out_dim,
                 out_init,
                 left_hidden_size      = 64,
                 left_hidden_depth     = 3,
                 left_out_size         = 64,
                 right_hidden_size     = 64,
                 right_hidden_depth    = 3,
                 right_out_size        = 64,
                 combined_hidden_size  = 128,
                 combined_hidden_depth = 2,
                 activation            = nn.ReLU(),
                 **kw_args):
        """
            A class defining a customizable "split" network, where we
            split the network into 2 halves before joining them into
            a "merged" section. The idea here comes from arXiv:1610.05182v1,
            although this implementation more closely resembles the network
            used in arXiv:1707.02286v2. In both cases, the goal is to
            split proprioceptive/egocentric information (think joints,
            angular velocities, etc.) from exteroceptive information that
            relates to the environment (terrain sensors, position in relation
            to the terrain, etc.).

            Arguments:

             in_dim                The dimensions of the input data. For
                                   instance, if the expected input shape is
                                   (batch_size, 16, 4), in_dim would be (16, 4).
             out_dim               The expected dimensions for the output. For
                                   instance, if the expected output shape is
                                   (batch_size, 16, 4), out_dim would be
                                   (16, 4).
             out_init              A std weight to apply to the output layer.
             left_hidden_size      The number of output neurons for each hidden
                                   layer of the left network. Note that this can
                                   be set to 0, resulting in only an input and
                                   output layer.
             left_hidden_depth     The number of hidden layers in the left
                                   network.
             left_out_size         The number of output neurons for the left
                                   network.
             right_hidden_size     The number of output neurons for each hidden
                                   layer of the right network.
             right_hidden_depth    The number of hidden layers in the right
                                   network.
             right_out_size        The number of output neurons for the right
                                   network.

             combined_hidden_size  The number of output neurons for each hidden
                                   layer of the right network. Note that this
                                   can be set to 0, resulting in only an input
                                   and output layer.
             combined_hidden_depth The number of hidden layers in the right
                                   network.
             activation            The activation function to use on the output
                                   of hidden layers.
        """


        super(SplitObsNetwork, self).__init__(
            out_dim = out_dim,
            **kw_args)

        if type(out_dim) == tuple:
            out_size     = reduce(lambda a, b: a*b, out_dim)
            self.out_dim = out_dim
        else:
            out_size     = out_dim
            self.out_dim = (out_dim,)

        self.activation = activation

        side_1_dim = self.split_start
        side_2_dim = in_dim - self.split_start

        # TODO: in the orignal paper, there is a "low level" section and
        # a "high level" section. The low level section handles
        # proprioceptive information (joints, positions, etc.), and the
        # high level section sees everything but only sends a signal every
        # K iterations. A later paper uses a similar technique, but it's
        # unclear if they keep the "K iteration" approach (they're very
        # vague). At any rate, this later approach is slighty different in
        # that it splits between proprioceptive and exteroceptive (sensory
        # information about the environment).
        #

        #
        # Left side network.
        #
        s1_kw_args = kw_args.copy()
        s1_kw_args["name"] = self.name + "_s1"

        self.s1_net = FeedForwardNetwork(
            in_dim       = side_1_dim,
            hidden_size  = left_hidden_size,
            hidden_depth = left_hidden_depth,
            out_dim      = left_out_size,
            activation   = self.activation,
            is_embedded  = True,
            **s1_kw_args)

        #
        # Right side network.
        #
        s2_kw_args = kw_args.copy()
        s2_kw_args["name"] = self.name + "_s2"

        self.s2_net = FeedForwardNetwork(
            in_dim       = side_2_dim,
            hidden_size  = right_hidden_size,
            hidden_depth = right_hidden_depth,
            out_dim      = right_out_size,
            activation   = self.activation,
            is_embedded  = True,
            **s2_kw_args)

        #
        # Combined sides network.
        #
        combined_in_size = left_out_size + right_out_size
        c_kw_args = kw_args.copy()
        c_kw_args["name"] = self.name + "_combined"

        self.combined_layers = FeedForwardNetwork(
            in_dim       = combined_in_size,
            hidden_size  = combined_hidden_size,
            hidden_depth = combined_hidden_depth,
            out_dim      = out_size,
            activation   = self.activation,
            is_embedded  = False,
            out_init     = out_init,
            **c_kw_args)

    def forward(self, _input):
        out = _input.flatten(start_dim = 1)

        s1_out = out[:, 0 : self.split_start]
        s2_out = out[:, self.split_start : ]

        #
        # Side 1 (left side).
        #
        s1_out = self.s1_net(s1_out)

        #
        # Side 2 (right side).
        #
        s2_out = self.s2_net(s2_out)

        #
        # Full layers.
        #
        out = torch.cat((s1_out, s2_out), dim=1)
        out = self.combined_layers(out)

        return out

class AtariPixelNetwork(PPOConv2dNetwork):

    def __init__(self,
                 in_shape,
                 out_dim,
                 out_init,
                 activation = nn.ReLU(),
                 **kw_args):

        super(AtariPixelNetwork, self).__init__(
            out_dim = out_dim,
            **kw_args)

        self.a_f = activation

        channels   = in_shape[0]
        height     = in_shape[1]
        width      = in_shape[2]

        k_s  = 8
        strd = 4
        pad  = 0
        self.conv1 = init_layer(nn.Conv2d(channels, 32,
            kernel_size=k_s, stride=strd, padding=pad))

        height = get_conv2d_out_size(height, pad, k_s, strd)
        width  = get_conv2d_out_size(width, pad, k_s, strd)

        k_s  = 4
        strd = 2
        pad  = 0
        self.conv2 = init_layer(nn.Conv2d(32, 64,
            kernel_size=k_s, stride=strd, padding=pad))

        height = get_conv2d_out_size(height, pad, k_s, strd)
        width  = get_conv2d_out_size(width, pad, k_s, strd)

        k_s  = 3
        strd = 1
        pad  = 0
        self.conv3 = init_layer(nn.Conv2d(64, 64,
            kernel_size=k_s, stride=strd, padding=pad))

        height = get_conv2d_out_size(height, pad, k_s, strd)
        width  = get_conv2d_out_size(width, pad, k_s, strd)

        self.l1 = init_layer(nn.Linear(height * width * 64, 512))
        self.l2 = init_layer(nn.Linear(512, out_dim), weight_std=out_init)


    def forward(self, _input):
        out = self.conv1(_input)
        out = self.a_f(out)

        out = self.conv2(out)
        out = self.a_f(out)

        out = self.conv3(out)
        out = self.a_f(out)

        out = out.flatten(start_dim=1)

        out = self.l1(out)
        out = self.a_f(out)

        out = self.l2(out)

        out = self.output_func(out)

        return out


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

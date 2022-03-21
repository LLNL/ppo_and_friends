import torch
import os
from torch import nn
import torch.nn.functional as F
from functools import reduce
import numpy as np
import sys
from .utils import *
from .ppo_networks import PPOActorCriticNetwork, PPOConv2dNetwork, SingleSplitObservationNetwork


class SimpleFeedForward(PPOActorCriticNetwork):

    def __init__(self,
                 #FIXME: change to in_size, out_size?
                 in_dim,
                 out_dim,
                 out_init     = None,
                 activation   = nn.ReLU(),
                 hidden_size  = 128,
                 hidden_depth = 2,
                 is_embedded  = False,
                 **kw_args):

        super(SimpleFeedForward, self).__init__(
            out_dim = out_dim,
            **kw_args)

        self.is_embedded = is_embedded

        if type(out_dim) == tuple:
            out_size     = reduce(lambda a, b: a*b, out_dim)
            self.out_dim = out_dim
        else:
            out_size     = out_dim
            self.out_dim = (out_dim,)

        self.activation  = activation
        self.input_layer = init_layer(nn.Linear(in_dim, hidden_size))

        hidden_layer_list = []
        for _ in range(hidden_depth):
            hidden_layer_list.append(init_layer(
                nn.Linear(
                    hidden_size,
                    hidden_size)))

            hidden_layer_list.append(self.activation)

        self.hidden_layers = nn.Sequential(*hidden_layer_list)

        if out_init != None:
            self.output_layer = init_layer(nn.Linear(hidden_size, out_size),
                weight_std=out_init)
        else:
            self.output_layer = init_layer(nn.Linear(hidden_size, out_size))

    def forward(self, _input):

        out = _input.flatten(start_dim = 1)

        out = self.input_layer(out)
        out = self.activation(out)

        out = self.hidden_layers(out)

        out = self.output_layer(out)

        #
        # If this network is embedded in a larger network,
        # we want to treat it as such and return the output
        # after activation.
        #
        if self.is_embedded:
            return self.activation(out)

        if self.need_softmax:
            out = F.softmax(out, dim=-1)

        out_shape = (out.shape[0],) + self.out_dim
        out = out.reshape(out_shape)

        return out


class SimpleSplitObsNetwork(SingleSplitObservationNetwork):

    def __init__(self,
                 in_dim,
                 out_dim,
                 out_init,
                 hidden_left_size   = 64,
                 hidden_right_size  = 64,
                 hidden_left_depth  = 2,
                 hidden_right_depth = 2,
                 output_depth       = 1,
                 activation         = nn.ReLU(),
                 **kw_args):

        super(SimpleSplitObsNetwork, self).__init__(
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

        self.s1_net = SimpleFeedForward(
            in_dim      = side_1_dim,
            hidden_size = hidden_left_size,
            out_dim     = hidden_left_size,
            activation  = self.activation,
            is_embedded = True,
            **s1_kw_args)

        #
        # Right side network.
        #
        s2_kw_args = kw_args.copy()
        s2_kw_args["name"] = self.name + "_s2"

        self.s2_net = SimpleFeedForward(
            in_dim      = side_2_dim,
            hidden_size = hidden_right_size,
            out_dim     = hidden_right_size,
            activation  = self.activation,
            is_embedded = True,
            **s2_kw_args)
        #
        # Combined sides network.
        #
        combined_hidden_size = hidden_left_size + hidden_right_size

        self.combined_layers = SimpleFeedForward(
            in_dim      = combined_hidden_size,
            hidden_size = combined_hidden_size,
            out_dim     = out_size,
            activation  = self.activation,
            is_embedded = False,
            out_init    = out_init,
            **s2_kw_args)

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

        if self.need_softmax:
            out = F.softmax(out, dim=-1)

        out_shape = (out.shape[0],) + self.out_dim
        out = out.reshape(out_shape)

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

        if self.need_softmax:
            out = F.softmax(out, dim=-1)

        return out

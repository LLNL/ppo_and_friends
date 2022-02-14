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
                 in_dim,
                 out_dim,
                 out_init,
                 activation   = nn.ReLU(),
                 hidden_size  = 128,
                 **kw_args):

        super(SimpleFeedForward, self).__init__(
            out_dim = out_dim,
            **kw_args)

        if type(out_dim) == tuple:
            out_size     = reduce(lambda a, b: a*b, out_dim)
            self.out_dim = out_dim
        else:
            out_size     = out_dim
            self.out_dim = (out_dim,)

        self.activation = activation

        self.l1 = init_layer(nn.Linear(in_dim, hidden_size))
        self.l2 = init_layer(nn.Linear(hidden_size, hidden_size))
        self.l3 = init_layer(nn.Linear(hidden_size, hidden_size))
        self.l4 = init_layer(nn.Linear(hidden_size, out_size),
            weight_std=out_init)

    def forward(self, _input):

        out = _input.flatten(start_dim = 1)

        out = self.l1(out)
        out = self.activation(out)

        out = self.l2(out)
        out = self.activation(out)

        out = self.l3(out)
        out = self.activation(out)

        out = self.l4(out)

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
                 hidden_left    = 64,
                 hidden_right   = 64,
                 num_out_layers = 1,
                 activation     = nn.ReLU(),
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
        s1_kw_args = kw_args.copy()
        s1_kw_args["name"] = self.name + "_s1"

        self.s1_net = SimpleFeedForward(
            in_dim     = side_1_dim,
            out_dim    = hidden_left,
            out_init   = np.sqrt(2),
            activation = self.activation,
            **s1_kw_args)

        s2_kw_args = kw_args.copy()
        s2_kw_args["name"] = self.name + "_s2"

        self.s2_net = SimpleFeedForward(
            in_dim     = side_2_dim,
            out_dim    = hidden_right,
            out_init   = np.sqrt(2),
            activation = self.activation,
            **s2_kw_args)

        inner_hidden_size  = hidden_left + hidden_right

        out_layer_list = []
        for _ in range(num_out_layers - 1):
            out_layer_list.append(init_layer(
                nn.Linear(
                    inner_hidden_size,
                    inner_hidden_size)))

            out_layer_list.append(activation)

        out_layer_list.append(init_layer(nn.Linear(inner_hidden_size,
            out_size), weight_std=out_init))

        self.out_layers = nn.Sequential(*out_layer_list)

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
        out = self.out_layers(out)

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

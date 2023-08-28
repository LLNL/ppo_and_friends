"""
    A home for observation encoders.
"""
import torch
import torch.nn as nn
from .utils import *
from ppo_and_friends.utils.misc import get_size_and_shape

class LinearObservationEncoder(nn.Module):

    def __init__(self,
                 obs_shape,
                 encoded_dim,
                 out_init,
                 hidden_size,
                 activation = nn.ReLU(),
                 **kw_args):
        """
            A simple encoder for encoding observations into
            forms that only contain information needed by the
            actor. In other words, we want to teach this model
            to get rid of any noise that may exist in the observation.
            By noise, we mean anything that does not pertain to
            the actions being taken.

            This implementation uses a simple feed-forward network.
        """

        super(LinearObservationEncoder, self).__init__()

        self.activation = activation

        obs_size, _ = get_size_and_shape(obs_shape)

        self.enc_1 = init_layer(nn.Linear(obs_size, hidden_size))
        self.enc_2 = init_layer(nn.Linear(hidden_size, hidden_size))
        self.enc_3 = init_layer(nn.Linear(hidden_size, hidden_size))
        self.enc_4 = init_layer(nn.Linear(hidden_size, encoded_dim),
            gain=out_init)

    def forward(self,
                obs):

        obs = obs.flatten(start_dim = 1)

        enc_obs = self.enc_1(obs)
        enc_obs = self.activation(enc_obs)

        enc_obs = self.enc_2(enc_obs)
        enc_obs = self.activation(enc_obs)

        enc_obs = self.enc_3(enc_obs)
        enc_obs = self.activation(enc_obs)

        enc_obs = self.enc_4(enc_obs)
        return enc_obs


class Conv2dObservationEncoder(nn.Module):

    def __init__(self,
                 obs_shape,
                 encoded_dim,
                 out_init,
                 activation = nn.ReLU(),
                 **kw_args):
        """
            A simple encoder for encoding observations into
            forms that only contain information needed by the
            actor. In other words, we want to teach this model
            to get rid of any noise that may exist in the observation.
            By noise, we mean anything that does not pertain to
            the actions being taken.

            This implementation uses 2d convolutions followed by
            linear layers.
        """

        super(Conv2dObservationEncoder, self).__init__()

        self.activation = activation

        channels = obs_shape[0]
        height   = obs_shape[1]
        width    = obs_shape[2]

        k_s  = 3
        pad  = 0
        strd = 1
        self.conv_1 = init_layer(nn.Conv2d(channels, 8,
            kernel_size=5, stride=1))
        height      = get_conv2d_out_size(height, pad, k_s, strd)
        width       = get_conv2d_out_size(width, pad, k_s, strd)

        k_s  = 3
        pad  = 0
        strd = 1
        self.mp_1 = nn.MaxPool2d(kernel_size=k_s, padding=pad, stride=strd)
        height    = get_maxpool2d_out_size(height, pad, k_s, strd)
        width     = get_maxpool2d_out_size(width, pad, k_s, strd)

        k_s  = 3
        pad  = 0
        strd = 1
        self.conv_2 = init_layer(nn.Conv2d(16, 16, kernel_size=5, stride=1))
        height      = get_conv2d_out_size(height, pad, k_s, strd)
        width       = get_conv2d_out_size(width, pad, k_s, strd)

        k_s  = 3
        pad  = 0
        strd = 1
        self.mp_2 = nn.MaxPool2d(kernel_size=k_s, padding=pad, stride=strd)
        height    = get_maxpool2d_out_size(height, pad, k_s, strd)
        width     = get_maxpool2d_out_size(width, pad, k_s, strd)

        k_s  = 3
        pad  = 0
        strd = 1
        self.conv_3 = init_layer(nn.Conv2d(16, 16, kernel_size=5, stride=1))
        height      = get_conv2d_out_size(height, pad, k_s, strd)
        width       = get_conv2d_out_size(width, pad, k_s, strd)

        k_s  = 3
        pad  = 0
        strd = 1
        self.mp_3 = nn.MaxPool2d(kernel_size=k_s, padding=pad, stride=strd)
        height    = get_maxpool2d_out_size(height, pad, k_s, strd)
        width     = get_maxpool2d_out_size(width, pad, k_s, strd)

        self.linear_encoder = LinearObservationEncoder(
            height * width * 16,
            encoded_dim,
            out_init,
            encoded_dim)


    def forward(self,
                obs):

        enc_obs = self.conv_1(obs)
        enc_obs = self.mp_1(enc_obs)
        enc_obs = self.activation(enc_obs)

        enc_obs = self.conv_2(enc_obs)
        enc_obs = self.mp_2(enc_obs)
        enc_obs = self.activation(enc_obs)

        enc_obs = self.conv_3(enc_obs)
        enc_obs = self.mp_3(enc_obs)
        enc_obs = self.activation(enc_obs)

        enc_obs = enc_obs.flatten(start_dim = 1)

        enc_obs = self.linear_encoder(enc_obs)

        return enc_obs


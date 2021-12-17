import torch
import os
from torch import nn
import torch.nn.functional as F
import numpy as np

class PPONetwork(nn.Module):

    def __init__(self):
        super(PPONetwork, self).__init__()
        self.name = "PPONetwork"

    def save(self, path):
        out_f = os.path.join(path, self.name + ".model")
        torch.save(self.state_dict(), out_f)

    def load(self, path):
        in_f = os.path.join(path, self.name + ".model")
        self.load_state_dict(torch.load(in_f))


class PPOConv2dNetwork(PPONetwork):

    def __init__(self):
        super(PPOConv2dNetwork, self).__init__()
        self.name = "PPOConv2dNetwork"


def get_conv2d_out_size(in_size,
                        padding,
                        kernel_size,
                        stride):
        out_size = int(((in_size + 2.0 * padding - (kernel_size - 1) - 1)\
            / stride) + 1)
        return out_size

def get_maxpool2d_out_size(in_size,
                           padding,
                           kernel_size,
                           stride):
    return get_conv2d_out_size(in_size, padding, kernel_size, stride)


########################################################################
#                        Actor Critic Networks                         #
########################################################################


class SimpleFeedForward(PPONetwork):

    def __init__(self,
                 name,
                 in_dim,
                 out_dim,
                 need_softmax = False):

        super(SimpleFeedForward, self).__init__()
        self.name = name
        self.need_softmax = need_softmax

        self.l1 = nn.Linear(in_dim, 128)
        self.l2 = nn.Linear(128, 128)
        self.l3 = nn.Linear(128, 128)
        self.l4 = nn.Linear(128, out_dim)

        self.l_relu = torch.nn.LeakyReLU()

    def forward(self, _input):

        out = self.l1(_input)
        out = self.l_relu(out)

        out = self.l2(out)
        out = self.l_relu(out)

        out = self.l3(out)
        out = self.l_relu(out)

        out = self.l4(out)

        if self.need_softmax:
            out = F.softmax(out, dim=-1)

        return out


class AtariRAMNetwork(PPONetwork):

    def __init__(self,
                 name,
                 in_dim,
                 out_dim,
                 need_softmax = False):

        super(AtariRAMNetwork, self).__init__()
        self.name   = name
        self.need_softmax = need_softmax
        self.a_f = torch.nn.ReLU()

        self.l1 = nn.Linear(in_dim, 1024)
        self.bn1 = nn.BatchNorm1d(1024)

        self.l2 = nn.Linear(1024, 512)
        self.bn2 = nn.BatchNorm1d(512)

        self.l3 = nn.Linear(512, 256)
        self.bn3 = nn.BatchNorm1d(256)

        self.l4 = nn.Linear(256, 128)
        self.bn4 = nn.BatchNorm1d(128)

        self.l5 = nn.Linear(128, 64)
        self.bn5 = nn.BatchNorm1d(64)

        self.l6 = nn.Linear(64, 32)
        self.bn6 = nn.BatchNorm1d(32)

        self.l7 = nn.Linear(32, 16)
        self.bn7 = nn.BatchNorm1d(16)

        self.l8 = nn.Linear(16, 8)
        self.bn8 = nn.BatchNorm1d(8)

        self.l9 = nn.Linear(8, out_dim)


    def forward(self, _input):

        out = self.l1(_input)
        out = self.bn1(out)
        out = self.a_f(out)

        out = self.l2(out)
        out = self.bn2(out)
        out = self.a_f(out)

        out = self.l3(out)
        out = self.bn3(out)
        out = self.a_f(out)

        out = self.l4(out)
        out = self.bn4(out)
        out = self.a_f(out)

        out = self.l5(out)
        out = self.bn5(out)
        out = self.a_f(out)

        out = self.l6(out)
        out = self.bn6(out)
        out = self.a_f(out)

        out = self.l7(out)
        out = self.bn7(out)
        out = self.a_f(out)

        out = self.l8(out)
        out = self.bn8(out)
        out = self.a_f(out)

        out = self.l9(out)

        if self.need_softmax:
            out = F.softmax(out, dim=-1)

        return out


#class AtariPixelNetwork(PPOConv2dNetwork):
#
#    def __init__(self,
#                 name,
#                 in_shape,
#                 out_dim,
#                 need_softmax = False):
#
#        super(AtariPixelNetwork, self).__init__()
#
#        self.name         = name
#        self.need_softmax = need_softmax
#        self.a_f          = torch.nn.ReLU()
#
#        height     = in_shape[0]
#        width      = in_shape[1]
#        channels   = in_shape[2]
#
#        k_s  = 8
#        strd = 4
#        pad  = 1
#        self.conv1 = nn.Conv2d(channels, 32,
#            kernel_size=k_s, stride=strd, padding=pad)
#        self.mp1   = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
#        self.bn1   = nn.BatchNorm2d(32)
#
#        height     = get_conv2d_out_size(height, pad, k_s, strd)
#        width      = get_conv2d_out_size(width, pad, k_s, strd)
#        height     = get_maxpool2d_out_size(height, 1, 3, 1)
#        width      = get_maxpool2d_out_size(width, 1, 3, 1)
#
#        k_s  = 5
#        strd = 3
#        pad  = 2
#        self.conv2 = nn.Conv2d(32, 32,
#            kernel_size=k_s, stride=strd, padding=pad)
#        self.mp2   = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
#        self.bn2   = nn.BatchNorm2d(32)
#
#        height     = get_conv2d_out_size(height, pad, k_s, strd)
#        width      = get_conv2d_out_size(width, pad, k_s, strd)
#        height     = get_maxpool2d_out_size(height, 1, 3, 1)
#        width      = get_maxpool2d_out_size(width, 1, 3, 1)
#
#        k_s  = 4
#        strd = 2
#        pad  = 2
#        self.conv3 = nn.Conv2d(32, 32,
#            kernel_size=k_s, stride=strd, padding=pad)
#        self.mp3   = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
#        self.bn3   = nn.BatchNorm2d(32)
#
#        height     = get_conv2d_out_size(height, pad, k_s, strd)
#        width      = get_conv2d_out_size(width, pad, k_s, strd)
#        height     = get_maxpool2d_out_size(height, 1, 3, 1)
#        width      = get_maxpool2d_out_size(width, 1, 3, 1)
#
#        #k_s  = 3
#        #strd = 2
#        #pad  = 2
#        #self.conv4 = nn.Conv2d(32, 32,
#        #    kernel_size=k_s, stride=strd, padding=pad)
#        #self.mp4   = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
#        #self.bn4   = nn.BatchNorm2d(32)
#
#        #height     = get_conv2d_out_size(height, pad, k_s, strd)
#        #width      = get_conv2d_out_size(width, pad, k_s, strd)
#        #height     = get_maxpool2d_out_size(height, 1, 3, 1)
#        #width      = get_maxpool2d_out_size(width, 1, 3, 1)
#
#        #k_s  = 3
#        #strd = 2
#        #pad  = 2
#        #self.conv5 = nn.Conv2d(32, 32, kernel_size=k_s,
#        #    stride=strd, padding=pad)
#        #self.mp5   = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
#        #self.bn5   = nn.BatchNorm2d(32)
#
#        #height     = get_conv2d_out_size(height, pad, k_s, strd)
#        #width      = get_conv2d_out_size(width, pad, k_s, strd)
#        #height     = get_maxpool2d_out_size(height, 1, 3, 1)
#        #width      = get_maxpool2d_out_size(width, 1, 3, 1)
#
#        self.l1 = nn.Linear(height * width * 32, 1024)
#        self.l2 = nn.Linear(1024, 256)
#        self.l3 = nn.Linear(256, out_dim)
#
#
#    def forward(self, _input):
#        out = self.conv1(_input)
#        out = self.mp1(out)
#        out = self.bn1(out)
#        out = self.a_f(out)
#
#        out = self.conv2(out)
#        out = self.mp2(out)
#        out = self.bn2(out)
#        out = self.a_f(out)
#
#        out = self.conv3(out)
#        out = self.mp3(out)
#        out = self.bn3(out)
#        out = self.a_f(out)
#
#        #out = self.conv4(out)
#        #out = self.mp4(out)
#        #out = self.bn4(out)
#        #out = self.a_f(out)
#
#        #out = self.conv5(out)
#        #out = self.mp5(out)
#        #out = self.bn5(out)
#        #out = self.a_f(out)
#
#        out = out.flatten(start_dim=1)
#
#        out = self.l1(out)
#        out = self.a_f(out)
#
#        out = self.l2(out)
#        out = self.a_f(out)
#
#        out = self.l3(out)
#
#        if self.need_softmax:
#            out = F.softmax(out, dim=-1)
#
#        return out


class AtariPixelNetwork(PPOConv2dNetwork):

    def __init__(self,
                 name,
                 in_shape,
                 out_dim,
                 need_softmax = False):

        super(AtariPixelNetwork, self).__init__()

        self.name         = name
        self.need_softmax = need_softmax
        self.a_f          = torch.nn.ReLU()

        height     = in_shape[0]
        width      = in_shape[1]
        channels   = in_shape[2]

        k_s  = 5
        strd = 2
        pad  = 1
        self.conv1 = nn.Conv2d(channels, 32,
            kernel_size=k_s, stride=strd, padding=pad)
        self.mp1   = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)

        height     = get_conv2d_out_size(height, pad, k_s, strd)
        width      = get_conv2d_out_size(width, pad, k_s, strd)
        height     = get_maxpool2d_out_size(height, 1, 3, 1)
        width      = get_maxpool2d_out_size(width, 1, 3, 1)

        k_s  = 5
        strd = 2
        pad  = 1
        self.conv2 = nn.Conv2d(32, 32,
            kernel_size=k_s, stride=strd, padding=pad)
        self.mp2   = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)

        height     = get_conv2d_out_size(height, pad, k_s, strd)
        width      = get_conv2d_out_size(width, pad, k_s, strd)
        height     = get_maxpool2d_out_size(height, 1, 3, 1)
        width      = get_maxpool2d_out_size(width, 1, 3, 1)

        k_s  = 5
        strd = 2
        pad  = 1
        self.conv3 = nn.Conv2d(32, 32,
            kernel_size=k_s, stride=strd, padding=pad)
        self.mp3   = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)

        height     = get_conv2d_out_size(height, pad, k_s, strd)
        width      = get_conv2d_out_size(width, pad, k_s, strd)
        height     = get_maxpool2d_out_size(height, 1, 3, 1)
        width      = get_maxpool2d_out_size(width, 1, 3, 1)

        self.l1 = nn.Linear(height * width * 32, 1024)
        self.l2 = nn.Linear(1024, 256)
        self.l3 = nn.Linear(256, out_dim)


    def forward(self, _input):
        out = self.conv1(_input)
        out = self.mp1(out)
        out = self.a_f(out)

        out = self.conv2(out)
        out = self.mp2(out)
        out = self.a_f(out)

        out = self.conv3(out)
        out = self.mp3(out)
        out = self.a_f(out)

        out = out.flatten(start_dim=1)

        out = self.l1(out)
        out = self.a_f(out)

        out = self.l2(out)
        out = self.a_f(out)

        out = self.l3(out)

        if self.need_softmax:
            out = F.softmax(out, dim=-1)

        return out


########################################################################
#                           ICM Networks                               #
########################################################################


#
# Linear setup.
#
class LinearObservationEncoder(nn.Module):

    def __init__(self,
                 obs_dim,
                 encoded_dim,
                 hidden_size):
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

        self.l_relu = nn.LeakyReLU()

        self.enc_1  = nn.Linear(obs_dim, hidden_size)
        self.enc_2  = nn.Linear(hidden_size, hidden_size)
        self.enc_3  = nn.Linear(hidden_size, hidden_size)
        self.enc_4  = nn.Linear(hidden_size, encoded_dim)

    def forward(self,
                obs):

        enc_obs = self.enc_1(obs)
        enc_obs = self.l_relu(enc_obs)

        enc_obs = self.enc_2(enc_obs)
        enc_obs = self.l_relu(enc_obs)

        enc_obs = self.enc_3(enc_obs)
        enc_obs = self.l_relu(enc_obs)

        enc_obs = self.enc_4(enc_obs)
        return enc_obs


class LinearInverseModel(nn.Module):

    def __init__(self,
                 in_dim,
                 out_dim,
                 hidden_size):
        """
            An implementation of the inverse model for the ICM method. The
            inverse model learns to predict the action that was performed
            when given the current and next states. States may also be
            observations, and the observations are typically encoded into
            a form that only contains information that the actor needs to
            make choices.

            This implementation uses a simple feed-forward network.
        """

        super(LinearInverseModel, self).__init__()

        self.l_relu = nn.LeakyReLU()

        #
        # Inverse model; Predict the a_1 given s_1 and s_2.
        #
        self.inv_1 = nn.Linear(in_dim, hidden_size)
        self.inv_2 = nn.Linear(hidden_size, hidden_size)
        self.inv_3 = nn.Linear(hidden_size, out_dim)

    def forward(self,
                enc_obs_1,
                enc_obs_2):

        obs_input = torch.cat((enc_obs_1, enc_obs_2), dim=1)

        out = self.inv_1(obs_input)
        out = self.l_relu(out)

        out = self.inv_2(out)
        out = self.l_relu(out)

        out = self.inv_3(out)
        out = F.softmax(out, dim=-1)

        return out


class LinearForwardModel(nn.Module):

    def __init__(self,
                 in_dim,
                 out_dim,
                 act_dim,
                 hidden_size,
                 action_type):
        """
            The forward model of the ICM method. In short, this learns to
            predict the changes in state given a current state and action.
            In reality, our states can be observations, and our observations
            are typically encoded into a form that only contains information
            that an actor needs to make decisions.

            This implementation uses a simple feed-forward network.
        """

        super(LinearForwardModel, self).__init__()

        self.l_relu      = nn.LeakyReLU()
        self.action_type = action_type
        self.act_dim     = act_dim

        #
        # Forward model; Predict s_2 given s_1 and a_1.
        #
        self.f_1 = nn.Linear(in_dim, hidden_size)
        self.f_2 = nn.Linear(hidden_size, hidden_size)
        self.f_3 = nn.Linear(hidden_size, out_dim)

    def forward(self,
                enc_obs_1,
                actions):

        actions = actions.flatten(start_dim = 1)

        if self.action_type == "discrete":
            actions = torch.nn.functional.one_hot(actions,
                num_classes=self.act_dim).float()

            #
            # One-hot adds an extra dimension. Let's get rid of that bit.
            #
            actions = actions.squeeze(-2)

        _input = torch.cat((enc_obs_1, actions), dim=1)

        #
        # Predict obs_2 given obs_1 and actions.
        #
        out = self.f_1(_input)
        out = self.l_relu(out)

        out = self.f_2(out)
        out = self.l_relu(out)

        out = self.f_3(out)
        return out


class LinearICM(PPONetwork):

    def __init__(self,
                 obs_dim,
                 act_dim,
                 action_type,
                 reward_scale = 0.01):
        """
            The Intrinsic Curiosit Model (ICM).
        """

        super(LinearICM, self).__init__()

        self.act_dim      = act_dim
        self.reward_scale = reward_scale
        self.action_type  = action_type

        self.l_relu       = nn.LeakyReLU()
        self.ce_loss      = nn.CrossEntropyLoss(reduction="mean")
        self.mse_loss     = nn.MSELoss(reduction="none")

        encoded_obs_dim   = 128
        hidden_dims       = 128

        #
        # Observation encoder.
        #
        self.obs_encoder = LinearObservationEncoder(
            obs_dim,
            encoded_obs_dim,
            hidden_dims)

        #
        # Inverse model; Predict the a_1 given s_1 and s_2.
        #
        self.inv_model = LinearInverseModel(
            encoded_obs_dim * 2, 
            act_dim,
            hidden_dims)

        #
        # Forward model; Predict s_2 given s_1 and a_1.
        #
        self.forward_model = LinearForwardModel(
            encoded_obs_dim + act_dim,
            encoded_obs_dim,
            act_dim,
            hidden_dims,
            action_type)

    def forward(self,
                obs_1,
                obs_2,
                actions):

        obs_1 = obs_1.flatten(start_dim = 1)
        obs_2 = obs_2.flatten(start_dim = 1)

        #
        # First, encode the observations.
        #
        enc_obs_1 = self.obs_encoder(obs_1)
        enc_obs_2 = self.obs_encoder(obs_2)

        #
        # Inverse model prediction.
        #
        action_pred = self.inv_model(enc_obs_1, enc_obs_2)

        if self.action_type == "discrete":
            inv_loss = self.ce_loss(action_pred, actions.flatten())
        elif self.action_type == "continuous":
            inv_loss = self.mse_loss(action_pred, actions).mean()

        #
        # Forward model prediction.
        #
        obs_2_pred = self.forward_model(enc_obs_1, actions)

        f_loss = self.mse_loss(obs_2_pred, enc_obs_2)

        intrinsic_reward = (self.reward_scale / 2.0) * f_loss.sum(dim=-1)
        f_loss           = 0.5 * f_loss.mean()

        return intrinsic_reward, inv_loss, f_loss


#
# Conv2d setup.
#
class Conv2dObservationEncoder(nn.Module):

    def __init__(self,
                 obs_dim,
                 encoded_dim,
                 hidden_size):
        """
            A simple encoder for encoding observations into
            forms that only contain information needed by the
            actor. In other words, we want to teach this model
            to get rid of any noise that may exist in the observation.
            By noise, we mean anything that does not pertain to
            the actions being taken.

            This implementation uses 2d convolutions followed by
            deconvolutions.
        """

        super(Conv2dObservationEncoder, self).__init__()

        self.l_relu = nn.LeakyReLU()

        #FIXME: need to figure out how to get this.
        num_channels = obs_dim

        self.enc_1 = nn.Conv2d(num_channels, 64, kernel_size=5, stride=1)
        self.enc_2 = nn.Conv2d(64, 64, kernel_size=5, stride=1)
        self.enc_3 = nn.Conv2d(64, 64, kernel_size=5, stride=1)

        self.enc_4 = nn.ConvTranspose2d(64, 64, kernel_size=5, stride=1)
        self.enc_5 = nn.ConvTranspose2d(64, 64, kernel_size=5, stride=1)
        self.enc_6 = nn.ConvTranspose2d(64, num_channels, kernel_size=5, stride=1)


    def forward(self,
                obs):

        enc_obs = self.enc_1(obs)
        enc_obs = self.l_relu(enc_obs)

        enc_obs = self.enc_2(enc_obs)
        enc_obs = self.l_relu(enc_obs)

        enc_obs = self.enc_3(enc_obs)
        enc_obs = self.l_relu(enc_obs)

        enc_obs = self.enc_4(enc_obs)
        enc_obs = self.l_relu(enc_obs)

        enc_obs = self.enc_5(enc_obs)
        enc_obs = self.l_relu(enc_obs)

        enc_obs = self.enc_6(enc_obs)

        return enc_obs

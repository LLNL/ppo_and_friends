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


class AtariROMNetwork(PPONetwork):

    def __init__(self,
                 name,
                 in_dim,
                 out_dim,
                 need_softmax = False):

        super(AtariROMNetwork, self).__init__()
        self.name   = name
        self.in_dim = in_dim
        self.need_softmax = need_softmax
        self.a_f = torch.nn.ReLU()

        l_out_f = lambda l_in, pad, ks, stride : \
            int(((l_in + 2 * pad - (ks - 1) - 1) / stride) + 1)

        self.conv1 = nn.Conv1d(1, 32, kernel_size=5, stride=2)
        l_out = l_out_f(in_dim, 0, 5, 2)

        self.conv2 = nn.Conv1d(32, 32, kernel_size=5, stride=2)
        l_out = l_out_f(l_out, 0, 5, 2)

        self.conv3 = nn.Conv1d(32, 32, kernel_size=5, stride=2)
        l_out = l_out_f(l_out, 0, 5, 2)

        self.l1    = nn.Linear(l_out * 32, 128)
        self.l2    = nn.Linear(128, 256)
        self.l3    = nn.Linear(256, 512)
        self.l4    = nn.Linear(512, 1024)
        self.l5    = nn.Linear(1024, out_dim)


    def forward(self, _input):

        out = _input.reshape((-1, 1, self.in_dim))

        out = self.conv1(out)
        out = self.a_f(out)

        out = self.conv2(out)
        out = self.a_f(out)

        out = self.conv3(out)
        out = self.a_f(out)

        #print(out.shape)
        out = out.flatten(start_dim = 1)

        out = self.l1(out)
        out = self.a_f(out)

        out = self.l2(out)
        out = self.a_f(out)

        out = self.l3(out)
        out = self.a_f(out)

        out = self.l4(out)
        out = self.a_f(out)

        out = self.l5(out)
        out = self.a_f(out)

        if self.need_softmax:
            out = F.softmax(out, dim=-1)

        return out




########################################################################
#                           ICM Networks                               #
########################################################################


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

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


class LinearNN(PPONetwork):

    def __init__(self,
                 name,
                 in_dim,
                 out_dim,
                 need_softmax = False):

        super(LinearNN, self).__init__()
        self.name = name
        self.need_softmax = need_softmax

        self.l1 = nn.Linear(in_dim, 64)
        self.l2 = nn.Linear(64, 64)
        self.l3 = nn.Linear(64, out_dim)


    def forward(self, _input):

        out = self.l1(_input)
        out = F.relu(out)

        out = self.l2(out)
        out = F.relu(out)

        out = self.l3(out)

        if self.need_softmax:
            out = F.softmax(out, dim=-1)

        return out


class LinearNN2(PPONetwork):

    def __init__(self,
                 name,
                 in_dim,
                 out_dim,
                 need_softmax = False):

        super(LinearNN2, self).__init__()
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


########################################################################
#                           ICM Networks                               #
########################################################################


class StateActionPredictor(PPONetwork):
    def __init__(self,
                 obs_dim,
                 act_dim):

        self.act_dim    = act_dim
        self.l_relu     = torch.nn.LeakyReLU()
        encoded_obs_dim = 128

        #
        # Observation encoder.
        #
        self.enc_1 = nn.Linear(obs_dim, 64)
        self.enc_2 = nn.Linear(64, 64)
        self.enc_3 = nn.Linear(64, 64)
        self.enc_4 = nn.Linear(64, encoded_obs_dim)

        #
        # Inverse model; Predict the a_1 given s_1 and s_2.
        #
        self.inv_1 = nn.Linear(encoded_obs_dim * 2, 256)
        self.inv_2 = nn.Linear(256, act_dim)

        #
        # Forward model; Predict s_2 given s_1 and a_1.
        #
        self.f_1 = nn.Linear(encoded_obs_dim + act_dim, 256)
        self.f_2 = nn.Linear(256, encoded_obs_dim)

    def forward(self,
                obs_1,
                obs_2,
                actions):

        action_pred = self.predict_actions(obs_1, obs_2)


    def encode_observation(self,
                           obs):

        enc_obs = self.enc_1(obs)
        enc_obs = self.l_relu(enc_obs)

        enc_obs = self.enc_2(enc_obs)
        enc_obs = self.l_relu(enc_obs)

        enc_obs = self.enc_3(enc_obs)
        enc_obs = self.l_relu(enc_obs)

        enc_obs = self.enc_4(enc_obs)
        return enc_obs

    def predict_actions(self,
                        obs_1,
                        obs_2):

        obs_1 = obs_1.flatten(start_dim = 1)
        obs_2 = obs_2.flatten(start_dim = 1)

        #
        # First, encode the observations.
        #
        enc_obs_1 = self.encode_observation(obs_1)
        enc_obs_2 = self.encode_observation(obs_2)

        obs_input = torch.cat((enc_obs_1, enc_obs_2), dim=1)

        #
        # Last, feed the encoded observations into the
        # inverse model for predicting the actions.
        #
        out = self.inv_1(obs_input)
        out = self.l_relu(out)

        out = self.inv_2(obs_input)
        out = self.softmax(out, dim=-1)

        return out

    def predict_observation(self,
                            obs_1,
                            actions):

        obs_1   = obs_1.flatten(start_dim = 1)
        actions = actions.flatten(start_dim = 1)
        actions = torch.nn.functional.one_hot(actions, nun_classes=self.act_dim)

        #
        # One-hot adds an extra dimension. Let's get rid of that bit.
        #
        actions = actions.squeeze(-2)

        #
        # First, encode the observation.
        #
        enc_obs_1 = self.encode_observation(obs_1)

        _input = torch.cat((enc_obs_1, actions), dim=1)

        #
        # Next, predict obs_2 given obs_1 and actions.
        #
        out = self.f_1(_input)
        out = self.l_relu(out)

        out = self.f_1(out)
        return out

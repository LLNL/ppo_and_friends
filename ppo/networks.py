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
                 act_dim,
                 action_type,
                 reward_scale = 0.01):

        super(StateActionPredictor, self).__init__()

        self.act_dim      = act_dim
        self.reward_scale = reward_scale
        self.action_type  = action_type

        self.l_relu       = nn.LeakyReLU()
        self.ce_loss      = nn.CrossEntropyLoss(reduction="none")
        self.mse_loss     = nn.MSELoss(reduction="none")

        encoded_obs_dim   = 128

        #
        # Observation encoder.
        #
        self.enc_1 = nn.Linear(obs_dim, 128)
        self.enc_2 = nn.Linear(128, 128)
        self.enc_3 = nn.Linear(128, 128)
        self.enc_4 = nn.Linear(128, encoded_obs_dim)

        #
        # Inverse model; Predict the a_1 given s_1 and s_2.
        #
        self.inv_1 = nn.Linear(encoded_obs_dim * 2, 128)
        self.inv_2 = nn.Linear(128, 128)
        self.inv_3 = nn.Linear(128, act_dim)

        #
        # Forward model; Predict s_2 given s_1 and a_1.
        #
        self.f_1 = nn.Linear(encoded_obs_dim + act_dim, 128)
        self.f_2 = nn.Linear(128, 128)
        self.f_3 = nn.Linear(128, encoded_obs_dim)

    def forward(self,
                obs_1,
                obs_2,
                actions):

        obs_1 = obs_1.flatten(start_dim = 1)
        obs_2 = obs_2.flatten(start_dim = 1)

        #
        # First, encode the observations.
        #
        enc_obs_1 = self.encode_observation(obs_1)
        enc_obs_2 = self.encode_observation(obs_2)

        #
        # Inverse model prediction.
        #
        action_pred = self.predict_actions(enc_obs_1, enc_obs_2)

        if self.action_type == "discrete":
            #FIXME: why not just allow mean reduction in loss?
            inv_loss = self.ce_loss(action_pred, actions.flatten())
            inv_loss = inv_loss.mean(dim=-1)
        else:
            inv_loss = self.mse_loss(action_pred, actions).mean()

        #
        # Forward model prediction.
        #
        obs_2_pred = self.predict_observation(enc_obs_1, actions)

        f_loss = self.mse_loss(obs_2_pred, enc_obs_2)

        intrinsic_reward = (self.reward_scale / 2.0) * f_loss.sum(dim=-1)
        f_loss           = 0.5 * f_loss.mean()

        return intrinsic_reward, inv_loss, f_loss


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

    def predict_observation(self,
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

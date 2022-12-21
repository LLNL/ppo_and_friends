"""
    A home for ICM network implementations.
"""
from .encoders import *
import torch
import torch.nn as nn
from functools import reduce
from .utils import init_layer
from .ppo_networks import PPONetwork
import torch.nn.functional as t_functional
from ppo_and_friends.utils.mpi_utils import rank_print

class LinearInverseModel(nn.Module):

    def __init__(self,
                 in_dim,
                 out_dim,
                 out_init,
                 hidden_size,
                 action_dtype,
                 activation = nn.ReLU(),
                 **kw_args):
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

        if type(out_dim) == tuple:
            out_size     = reduce(lambda a, b: a*b, out_dim)
            self.out_dim = out_dim
        else:
            out_size     = out_dim
            self.out_dim = (out_dim,)

        self.output_func = lambda x : x

        if action_dtype in ["discrete", "multi-discrete"]:
            self.output_func = lambda x : t_functional.softmax(x, dim=-1)

        elif action_dtype == "multi-binary":
            self.output_func = t_functional.sigmoid

        self.activation = activation

        #
        # Inverse model; Predict the a_1 given s_1 and s_2.
        #
        self.inv_1 = init_layer(nn.Linear(in_dim, hidden_size))
        self.inv_2 = init_layer(nn.Linear(hidden_size, hidden_size))
        self.inv_3 = init_layer(nn.Linear(hidden_size, out_size),
            weight_std=out_init)

    def forward(self,
                enc_obs_1,
                enc_obs_2):

        obs_input = torch.cat((enc_obs_1, enc_obs_2), dim=1)

        out = self.inv_1(obs_input)
        out = self.activation(out)

        out = self.inv_2(out)
        out = self.activation(out)

        out = self.inv_3(out)

        out = self.output_func(out)

        out_shape = (out.shape[0],) + self.out_dim
        out = out.reshape(out_shape)

        return out


class LinearForwardModel(nn.Module):

    def __init__(self,
                 in_dim,
                 out_dim,
                 out_init,
                 hidden_size,
                 action_dtype,
                 act_dim,
                 action_nvec = None,
                 activation  = nn.ReLU(),
                 **kw_args):
        """
            The forward model of the ICM method. In short, this learns to
            predict the changes in state given a current state and action.
            In reality, our states can be observations, and our observations
            are typically encoded into a form that only contains information
            that an actor needs to make decisions.

            This implementation uses a simple feed-forward network.
        """

        super(LinearForwardModel, self).__init__()

        if type(out_dim) == tuple:
            out_size = reduce(lambda a, b: a*b, out_dim)
        else:
            out_size = out_dim

        self.activation   = activation
        self.action_dtype = action_dtype
        self.act_dim      = act_dim
        self.action_nvec  = action_nvec

        #
        # Forward model; Predict s_2 given s_1 and a_1.
        #
        self.f_1 = init_layer(nn.Linear(in_dim, hidden_size))
        self.f_2 = init_layer(nn.Linear(hidden_size, hidden_size))
        self.f_3 = init_layer(nn.Linear(hidden_size, out_size),
            weight_std=out_init)

    def forward(self,
                enc_obs_1,
                actions):

        if self.action_dtype == "discrete":
            #
            # This is a bit funny here... ICM likes to use one-hot encodings of
            # the actions, but we also need to flatten the actions before we
            # can concat with our encoded observations. So, if we have a
            # multi-dimensional discrete action space, we can create a multi-
            # dimensional one-hot encoding, but we then need to flatten it.
            #
            actions = t_functional.one_hot(actions,
                num_classes=self.act_dim[-1]).float().flatten(start_dim = 1)

        elif self.action_dtype == "multi-discrete":
            #
            # In the multi-discrete case, we need to create multiple
            # one-hot encodings, and they can be "ragged" (different lengths).
            #
            one_hots = []
            start    = 0

            for idx, dim in enumerate(self.action_nvec):
                stop = start + dim

                one_hots.append(t_functional.one_hot(actions[:, start : stop],
                    num_classes=dim).float().flatten(start_dim = 1))

                start = stop

            actions = torch.cat(one_hots, dim=1)

        _input = torch.cat((enc_obs_1, actions), dim=1)

        #
        # Predict obs_2 given obs_1 and actions.
        #
        out = self.f_1(_input)
        out = self.activation(out)

        out = self.f_2(out)
        out = self.activation(out)

        out = self.f_3(out)
        return out


class ICM(PPONetwork):

    def __init__(self,
                 obs_dim,
                 act_dim,
                 action_dtype,
                 out_init        = 1.0,
                 obs_encoder     = LinearObservationEncoder,
                 reward_scale    = 0.01,
                 activation      = nn.ReLU(),
                 encoded_obs_dim = 128,
                 hidden_size     = 128,
                 action_nvec     = None,
                 **kw_args):
        """
            The Intrinsic Curiosit Model (ICM).

            This implementation of ICM comes from arXiv:1705.05363v1.
        """

        super(ICM, self).__init__(**kw_args)

        if type(act_dim) == tuple:
            act_size = reduce(lambda a, b: a*b, act_dim)
            act_dim  = act_dim
        else:
            act_size = act_dim
            act_dim  = (act_dim,)

        self.reward_scale = reward_scale
        self.action_dtype = action_dtype
        self.action_nvec  = action_nvec

        self.activation   = activation
        self.ce_loss      = nn.CrossEntropyLoss(reduction="mean")
        self.mse_loss     = nn.MSELoss(reduction="none")

        #
        # Observation encoder.
        #
        if encoded_obs_dim > 0:
            self.obs_encoder = obs_encoder(
                obs_dim,
                encoded_obs_dim,
                out_init,
                hidden_size,
                **kw_args)
        else:
            self.obs_encoder = lambda x : x
            encoded_obs_dim  = obs_dim

        #
        # Inverse model; Predict the a_1 given s_1 and s_2.
        #
        self.inv_model = LinearInverseModel(
            encoded_obs_dim * 2, 
            act_dim,
            out_init,
            hidden_size,
            action_dtype,
            **kw_args)

        #
        # Forward model; Predict s_2 given s_1 and a_1.
        #
        self.forward_model = LinearForwardModel(
            encoded_obs_dim + act_size,
            encoded_obs_dim,
            out_init,
            hidden_size,
            action_dtype,
            act_dim,
            action_nvec = action_nvec,
            **kw_args)

    def forward(self,
                obs_1,
                obs_2,
                actions):

        #
        # First, encode the observations.
        #
        enc_obs_1 = self.obs_encoder(obs_1)
        enc_obs_2 = self.obs_encoder(obs_2)

        #
        # Inverse model prediction.
        #
        action_pred = self.inv_model(enc_obs_1, enc_obs_2)

        if self.action_dtype in ["discrete", "multi-discrete"]:

            shape_len = len(action_pred.shape)

            if shape_len != 2 and shape_len != 3:
                msg  = "ERROR: encountered unexpected shape when "
                msg += "predicting actions in ICM: "
                msg += "{}.".format(action_pred.shape)
                rank_print(msg)
                comm.Abort()
            elif shape_len == 3:
                #
                # Our action space is (batch_size, dims, num_classes), but
                # torch requires shape (batch_size, num_classes, dims) for
                # cross entropy.
                #
                action_pred = action_pred.transpose(1, 2)

            if self.action_dtype == "multi-discrete":
                inv_loss = 0
                start    = 0

                #
                # In the multi-discrete case, we have multiple losses to
                # calculate. Let's just sum them all up.
                #
                for idx, dim in enumerate(self.action_nvec):
                   stop = start + dim
                   inv_loss += self.ce_loss(action_pred[:, start : stop],
                       actions[:, idx:idx+1].flatten())

                   start = stop
            else:
               inv_loss = self.ce_loss(action_pred, actions.squeeze(1))

        elif self.action_dtype == "continuous":
            actions = actions.reshape(action_pred.shape)
            inv_loss = self.mse_loss(action_pred, actions).mean()

        #
        # Forward model prediction.
        #
        obs_2_pred = self.forward_model(enc_obs_1, actions)
        f_loss = self.mse_loss(obs_2_pred, enc_obs_2)

        intrinsic_reward = (self.reward_scale / 2.0) * f_loss.sum(dim=-1)
        f_loss           = 0.5 * f_loss.mean()

        return intrinsic_reward, inv_loss, f_loss


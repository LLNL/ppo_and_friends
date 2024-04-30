"""
    A home for ICM network implementations.
"""
from ppo_and_friends.networks.encoders import *
import torch
import torch.nn as nn
from functools import reduce
from ppo_and_friends.networks.utils import init_layer, create_sequential_network
from ppo_and_friends.utils.misc import get_size_and_shape
from ppo_and_friends.networks.ppo_networks.base import PPONetwork
import torch.nn.functional as t_functional
from ppo_and_friends.utils.mpi_utils import rank_print
from ppo_and_friends.utils.misc import get_flattened_space_length
from ppo_and_friends.utils.misc import get_space_dtype_str, get_space_shape, get_action_prediction_shape


from mpi4py import MPI
comm      = MPI.COMM_WORLD
rank      = comm.Get_rank()
num_procs = comm.Get_size()

class LinearInverseModel(nn.Module):

    def __init__(self,
                 in_size,
                 out_shape,
                 out_init,
                 hidden_size,
                 hidden_depth,
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

            Arguments:
                in_size         The size of the input (int).
                out_shape       The expected shape for the output. For
                                instance, if the expected output shape is
                                (batch_size, 16, 4), out_shape would be (16, 4).
                out_init        A std weight to apply to the output layer.
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
                action_dtype    A string signifying what type of actions we're
                                using.
                activation      The activation function to use on the output
                                of hidden layers.
        """

        super(LinearInverseModel, self).__init__()

        if type(out_shape) == tuple:
            out_size       = reduce(lambda a, b: a*b, out_shape)
            self.out_shape = out_shape
        else:
            out_size       = out_shape
            self.out_shape = (out_shape,)

        self.output_func = lambda x : x

        if action_dtype in ["discrete", "multi-discrete"]:
            self.output_func = lambda x : t_functional.softmax(x, dim=-1)

        elif action_dtype == "multi-binary":
            self.output_func = t_functional.sigmoid

        elif action_dtype != "continuous":
            rank_print(f"ERROR: unsupported action space {action_dtype}")
            comm.Abort()

        self.activation = activation

        #
        # Inverse model; Predict the a_1 given s_1 and s_2.
        #
        self.sequential_net = \
            create_sequential_network(
                in_size      = in_size,
                out_size     = out_size,
                hidden_size  = hidden_size,
                hidden_depth = hidden_depth,
                activation   = activation,
                out_init     = out_init)

    def forward(self,
                enc_obs_1,
                enc_obs_2):

        obs_input = torch.cat((enc_obs_1, enc_obs_2), dim=1)

        out = self.sequential_net(obs_input)
        out = self.output_func(out)

        out_shape = (out.shape[0],) + self.out_shape
        out = out.reshape(out_shape)

        return out


class LinearForwardModel(nn.Module):

    def __init__(self,
                 in_size,
                 out_shape,
                 out_init,
                 hidden_size,
                 hidden_depth,
                 action_dtype,
                 act_shape,
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

            Arguments:
                in_size         The size of the input (int).
                out_shape       The expected shape for the output. For
                                instance, if the expected output shape is
                                (batch_size, 16, 4), out_shape would be (16, 4).
                out_init        A std weight to apply to the output layer.
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
                action_dtype    A string signifying what type of actions we're
                                using.
                activation      The activation function to use on the output
                                of hidden layers.
        """

        super(LinearForwardModel, self).__init__()

        if type(out_shape) == tuple:
            out_size = reduce(lambda a, b: a*b, out_shape)
        else:
            out_size = out_shape

        self.activation   = activation
        self.action_dtype = action_dtype
        self.act_shape    = act_shape
        self.action_nvec  = action_nvec

        #
        # Forward model; Predict s_2 given s_1 and a_1.
        #
        self.sequential_net = \
            create_sequential_network(
                in_size      = in_size,
                out_size     = out_size,
                hidden_size  = hidden_size,
                hidden_depth = hidden_depth,
                activation   = activation,
                out_init     = out_init)

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
                num_classes=self.act_shape[-1]).float().flatten(start_dim = 1)

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
        out = self.sequential_net(_input)
        return out


class ICM(PPONetwork):

    def __init__(self,
                 obs_space,
                 action_space,
                 out_init             = 1.0,
                 obs_encoder          = LinearObservationEncoder,
                 reward_scale         = 0.01,
                 activation           = nn.ReLU(),
                 encoded_obs_dim      = 128,
                 encoder_hidden_size  = 128,
                 inverse_hidden_size  = 128,
                 inverse_hidden_depth = 2,
                 forward_hidden_size  = 128,
                 forward_hidden_depth = 2,
                 **kw_args):
        """
            The Intrinsic Curiosit Model (ICM).

            This implementation of ICM comes from arXiv:1705.05363v1.

            Arguments:
                out_init             A std weight to apply to the output layer.
                obs_encoder          The class to use for encoding observations.
                reward_scale         A scale/weight to apply to the reward.
                activation           The activation function to use on the
                                     output of hidden layers.
                encoded_obs_dim      The dimensions for the encoded
                                     observations.
                encoder_hidden_size  Hidden size for the encoder.
                                     Can either be an int or list of ints. If an
                                     int, all layers will be this size.
                                     Otherwise, a list designates the size for
                                     each layer. Note that the hidden_depth
                                     argument is ignored if this argument is a
                                     list and the depth is instead taken from
                                     the length of the list. Note that this 
                                     argument can be set to 0 or an empty list,
                                     resulting in only an input and output
                                     layer.
                inverse_hidden_size  Hidden size for the inverse model.
                                     Can either be an int or list of ints. If an
                                     int, all layers will be this size.
                                     Otherwise, a list designates the size for
                                     each layer. Note that the hidden_depth
                                     argument is ignored if this argument is a
                                     list and the depth is instead taken from
                                     the length of the list. Note that this 
                                     argument can be set to 0 or an empty list,
                                     resulting in only an input and output
                                     layer.
                inverse_hidden_depth The number of hidden layers for the inverse
                                     model. Note that this is ignored if
                                     hidden_size is a list.
                forward_hidden_size  Hidden size for the forward model.
                                     Can either be an int or list of ints. If an
                                     int, all layers will be this size.
                                     Otherwise, a list designates the size for
                                     each layer. Note that the hidden_depth
                                     argument is ignored if this argument is a
                                     list and the depth is instead taken from
                                     the length of the list. Note that this 
                                     argument can be set to 0 or an empty list,
                                     resulting in only an input and output
                                     layer.
                forward_hidden_depth The number of hidden layers for the forward
                                     model. Note that this is ignored if
                                     hidden_size is a list.
        """
        super(ICM, self).__init__(
            in_shape  = None,
            out_shape = None,
            **kw_args)

        self.obs_space    = obs_space
        self.action_space = action_space

        act_shape = get_action_prediction_shape(action_space)

        act_size, act_shape = get_size_and_shape(act_shape)

        self.reward_scale = reward_scale
        self.action_dtype = get_space_dtype_str(action_space)

        if self.action_dtype not in ["discrete", "multi-discrete", "continuous"]:
            msg  = f"ERROR: action type of {self.action_dtype} is not currenty "
            msg += "in the ICM module."
            rank_print(msg)
            comm.Abort()

        self.action_nvec = None
        if hasattr(action_space, "nvec"):
            self.action_nvec  = action_space.nvec

        self.activation   = activation
        self.ce_loss      = nn.CrossEntropyLoss(reduction="mean")
        self.mse_loss     = nn.MSELoss(reduction="none")

        #
        # Observation encoder.
        #
        if encoded_obs_dim > 0:
            obs_shape  = get_space_shape(obs_space)

            self.obs_encoder = obs_encoder(
                obs_shape,
                encoded_obs_dim,
                out_init,
                encoder_hidden_size,
                **kw_args)
        else:
            self.obs_encoder = lambda x : x
            encoded_obs_dim  = get_flattened_space_length(obs_space)

        #
        # Inverse model; Predict the a_1 given s_1 and s_2.
        #
        self.inv_model = LinearInverseModel(
            encoded_obs_dim * 2, 
            act_shape,
            out_init,
            inverse_hidden_size,
            inverse_hidden_depth,
            self.action_dtype,
            **kw_args)

        #
        # Forward model; Predict s_2 given s_1 and a_1.
        #
        self.forward_model = LinearForwardModel(
            encoded_obs_dim + act_size,
            encoded_obs_dim,
            out_init,
            forward_hidden_size,
            forward_hidden_depth,
            self.action_dtype,
            act_shape,
            action_nvec = self.action_nvec,
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


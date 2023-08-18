import torch
import torch.nn as nn
from torch.nn import functional as t_func
import math
import numpy as np
from ppo_and_friends.networks.ppo_networks.base import PPONetwork
from ppo_and_friends.networks.actor_critic.utils import get_actor_distribution
from ppo_and_friends.networks.attention import SelfAttentionEncodingBlock
from ppo_and_friends.networks.attention import SelfAttentionDecodingBlock
from ppo_and_friends.utils.misc import get_space_shape
from ppo_and_friends.utils.misc import get_action_dtype
from ppo_and_friends.utils.misc import get_flattened_space_length
from ppo_and_friends.networks.utils import init_layer

from mpi4py import MPI
comm      = MPI.COMM_WORLD
rank      = comm.Get_rank()
num_procs = comm.Get_size()


class MATActor(PPONetwork):

    def __init__(
        self,
        obs_space,
        action_space,
        num_agents,
        embedding_size           = 64,
        num_blocks               = 1,
        num_heads                = 1,
        internal_init            = nn.init.calculate_gain('relu'),
        out_init                 = 0.01,
        activation               = nn.GELU(),
        decoder_internal_init    = nn.init.calculate_gain('relu'),
        decoder_out_init         = 0.01,
        decoder_activation      = nn.GELU(),
        self_atten_internal_init = 0.01,
        self_atten_out_init      = 0.01,
        name                     = 'actor',
        **kw_args):
        """
        """
        super(MATActor, self).__init__(
            name      = name,
            in_shape  = (embedding_size,),
            out_shape = get_space_shape(action_space),
            **kw_args)

        self.obs_space = obs_space
        self.distribution, self.output_func = \
            get_actor_distribution(action_space, **kw_args)

        self.action_dtype   = get_action_dtype(action_space)
        self.embedding_size = embedding_size
        action_dim          = get_flattened_space_length(action_space)

        # TODO: let's update to support binary and multi-binary as well.
        supported_types = ["continuous", "discrete"]
        if self.action_dtype not in supported_types:
            msg  = "ERROR: you're using action space of type {self.action_dtype}, "
            msg += "but the multi-agent transformer currently only supports "
            msg += f"{supported_types}."
            rank_print(msg)
            comm.Abort()

        #
        # The original paper uses this "relu" gain
        #
        internal_gain = nn.init.calculate_gain('relu')

        if self.action_dtype == 'discrete':
            self.action_encoder = nn.Sequential(
                init_layer(nn.Linear(action_dim + 1, embedding_size, bias=False),
                    gain = internal_init),
                activation)
        else:
            self.action_encoder = nn.Sequential(
                init_layer(nn.Linear(action_dim, embedding_size),
                    gain = internal_init),
                activation)

        self.ln     = nn.LayerNorm(embedding_size)
        self.blocks = nn.Sequential(
            *[SelfAttentionDecodingBlock(
                  embedding_size,
                  num_heads,
                  num_agents,
                  activation               = decoder_activation,
                  internal_init            = decoder_internal_init,
                  out_init                 = decoder_out_init,
                  self_atten_internal_init = self_atten_internal_init,
                  self_atten_out_init      = self_atten_out_init)
              for _ in range(num_blocks)])

        self.head   = nn.Sequential(
            init_layer(nn.Linear(embedding_size, embedding_size),
                gain = internal_init),
            activation,
            nn.LayerNorm(embedding_size),
            init_layer(nn.Linear(embedding_size, action_dim),
                gain = out_init))

    def forward(self, action, encoded_obs):
        """
        """
        # action: (batch, num_agents, action_dim), one-hot/logits?
        # obs_rep: (batch, num_agents, embedding_size)
        #print(f"\nDecoder action shape: {action.shape}")#FIXME
        #print(f"\nDecoder action: {action}")#FIXME
        #Decoder action shape: torch.Size([N, 3, 6])
        # NOTE: I think what's happening here is we get a batch of action sequences as
        # (N, A, D) s.t. N is the batch size, A is `(num_agents - 1) + 1`, and D is
        # the action dimension + 1 for the start token. A is the sequence of actions so
        # so far (at most num_agents - 1) + the start token. We are predicting the action
        # for the next agent in line. Note that the action space is only expanded for
        # discrete actions! Continuous actions are the same dimensions.

        x = self.action_encoder(action)
        x = self.ln(x)

        for block in self.blocks:
            x = block(x, encoded_obs)

        x = self.head(x)
        x = self.output_func(x)
        return x

    def get_refined_prediction(self, action, encoded_obs):
        """
        Send an actor's predicted probabilities through its
        distribution's refinement method.
    
        Parameters
        ----------
        obs: gymnasium space
            The observation to infer from.
    
        Returns
        -------
        float 
            The predicted result sent through the distribution's
            refinement method.
        """
        res = self.__call__(action, encoded_obs)
        res = self.distribution.refine_prediction(res)
        return res


class MATCritic(PPONetwork):

    def __init__(self,
        obs_space,
        num_agents,
        embedding_size           = 64,
        num_blocks               = 1,
        num_heads                = 1,
        out_init                 = 0.01,
        internal_init            = nn.init.calculate_gain('relu'),
        activation               = nn.GELU(),
        encoder_internal_init    = nn.init.calculate_gain('relu'),
        encoder_out_init         = 0.01,
        encoder_activation       = nn.GELU(),
        self_atten_internal_init = 0.01,
        self_atten_out_init      = 0.01,
        name                     = 'critic',
        **kw_args):
        """
        """

        super(MATCritic, self).__init__(
            name      = name,
            in_shape  = get_space_shape(obs_space),
            out_shape = (1,),
            **kw_args)

        self.obs_space      = obs_space
        self.embedding_size = embedding_size
        self.num_agents     = num_agents

        self.obs_encoder = nn.Sequential(
            nn.LayerNorm(self.in_size),
            init_layer(nn.Linear(self.in_size, embedding_size),
                gain = internal_init),
            activation)

        self.ln     = nn.LayerNorm(embedding_size)
        self.blocks = nn.Sequential(
            *[SelfAttentionEncodingBlock(
                  embedding_size,
                  num_heads,
                  num_agents,
                  activation               = encoder_activation,
                  internal_init            = encoder_internal_init,
                  out_init                 = encoder_out_init,
                  self_atten_internal_init = self_atten_internal_init,
                  self_atten_out_init      = self_atten_out_init)
              for _ in range(num_blocks)])

        self.head   = nn.Sequential(
            init_layer(nn.Linear(embedding_size, embedding_size),
                gain = internal_init),
            activation,
            nn.LayerNorm(embedding_size),
            init_layer(nn.Linear(embedding_size, 1),
                gain = out_init))

    def encode_obs(self, obs):
        """
        """
        x = self.obs_encoder(obs)
        x = self.ln(x)
        return self.blocks(x)

    def forward(self, obs):
        encoded_obs = self.encode_obs(obs) 
        return self.head(encoded_obs)

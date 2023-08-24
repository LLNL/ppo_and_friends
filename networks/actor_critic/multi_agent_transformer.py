import torch
import torch.nn as nn
from torch.nn import functional as t_func
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
        **kw_args):
        """
        Parameters:
        -----------
        obs_space: gymnasium space
            The observation space for this actor.
        action_space: gymnasium space
            The action space for this actor.
        num_agents: int
            The number of agents we're training.
        embedding_size: int
            The size of the embedded/encoded observations.
        num_blocks: int
            The number of self-attention decoding blocks to use.
        num_heads: int
            The number of heads for each self-attention decoding block.
        internal_init: float
            Gain initialization for internal layers.
        out_init: float
            Gain initialization for the output layer.
        activation: activation function
            The activation function to use.
        decoder_internal_init: float
            Gain for the internal layers of the decoder.
        decoder_out_init: float
            Gain for the output layers of the decoder.
        decoder_activation: activation function
            The activation function for the decoder.
        self_atten_internal_init: float
            Gain for the internal self-attention layers.
        self_atten_out_init: float
            Gain for the output self-attention layer.
        """
        super(MATActor, self).__init__(
            name      = "mat_actor",
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
        **kw_args):
        """
        Parameters:
        -----------
        obs_space: gymnasium space
            The observation space for this critic.
        num_agents: int
            The number of agents we're training.
        embedding_size: int
            The size of the embedded/encoded observations.
        num_blocks: int
            The number of self-attention decoding blocks to use.
        num_heads: int
            The number of heads for each self-attention decoding block.
        internal_init: float
            Gain initialization for internal layers.
        out_init: float
            Gain initialization for the output layer.
        activation: activation function
            The activation function to use.
        encoder_internal_init: float
            Gain for the internal layers of the encoder.
        encoder_out_init: float
            Gain for the output layers of the encoder.
        encoder_activation: activation function
            The activation function for the encoder.
        self_atten_internal_init: float
            Gain for the internal self-attention layers.
        self_atten_out_init: float
            Gain for the output self-attention layer.
        """

        super(MATCritic, self).__init__(
            name      = "mat_critic",
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
        x = self.blocks(x)
        return x

    def forward(self, obs):
        encoded_obs = self.encode_obs(obs)
        return encoded_obs, self.head(encoded_obs)


class MATActorCritic(PPONetwork):

    def __init__(
        self,
        obs_space,
        action_space,
        num_agents,
        name = 'actor_critic',
        **kw_args):
        """
        Parameters:
        -----------
        obs_space: gymnasium space
            The observation space for this actor.
        action_space: gymnasium space
            The action space for this actor.
        num_agents: int
            The number of agents we're training.
        name: str
            The name of this class.
        """
        super(MATActorCritic, self).__init__(
            name      = name,
            in_shape  = None,
            out_shape = None,
            **kw_args)

        self.actor = MATActor(
            obs_space    = obs_space,
            action_space = action_space,
            num_agents   = num_agents,
            **kw_args)

        self.critic = MATCritic(
            obs_space    = obs_space,
            action_space = action_space,
            num_agents   = num_agents,
            **kw_args)

    def forward(self, obs, action_block):

        encoded_obs, values = self.critic(obs)
        action_pred         = self.actor(action_block, encoded_obs)
        return values, action_pred

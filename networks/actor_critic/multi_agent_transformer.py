import torch
import torch.nn as nn
from torch.nn import functional as t_func
import math
import numpy as np
from ppo_and_friends.networks.ppo_networks.base import PPONetwork
from ppo_and_friends.networks.actor_critic.utils import get_actor_distribution
from ppo_and_friends.utils.misc import get_space_shape
from ppo_and_friends.utils.misc import get_action_dtype
from ppo_and_friends.utils.misc import get_flattened_space_length

from mpi4py import MPI
comm      = MPI.COMM_WORLD
rank      = comm.Get_rank()
num_procs = comm.Get_size()

#FIXME: This is very similar to our method but a bit more configurable. Let's update ours.
def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    if module.bias is not None:
        bias_init(module.bias.data)
    return module

def init_(m, gain=0.01, activate=False):
    if activate:
        gain = nn.init.calculate_gain('relu')
    return init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), gain=gain)

#############################################################################
# FIXME: move SelfAttention and Encoder to some kind of utilities file. Maybe the encoders.py? It
# might be better to create an encoders directory that contains multiple styles of encoders.
class SelfAttention(nn.Module):

    def __init__(self, embedding_size, num_heads, num_agents, masked=False):
        """
        """
        super(SelfAttention, self).__init__()

        #FIXME: replace with MPI
        assert embedding_size % num_heads == 0

        self.masked = masked
        self.num_heads = num_heads

        # key, query, value projections for all heads
        #FIXME: replace with our init?
        self.key   = init_(nn.Linear(embedding_size, embedding_size))
        self.query = init_(nn.Linear(embedding_size, embedding_size))
        self.value = init_(nn.Linear(embedding_size, embedding_size))

        # output projection
        self.proj = init_(nn.Linear(embedding_size, embedding_size))

        # if self.masked:
        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer("mask", torch.tril(torch.ones(num_agents + 1, num_agents + 1))
                             .view(1, 1, num_agents + 1, num_agents + 1))

        #FIXME: what is this?? It looks like it's unused...
        self.att_bp = None

    def forward(self, key, value, query):
        #FIXME: update names
        #print(f"QUERY: {query.size()}")#FIXME
        B, L, D = query.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(key).view(B, L, self.num_heads, D // self.num_heads).transpose(1, 2)  # (B, nh, L, hs)
        q = self.query(query).view(B, L, self.num_heads, D // self.num_heads).transpose(1, 2)  # (B, nh, L, hs)
        v = self.value(value).view(B, L, self.num_heads, D // self.num_heads).transpose(1, 2)  # (B, nh, L, hs)

        # causal attention: (B, nh, L, hs) x (B, nh, hs, L) -> (B, nh, L, L)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))

        if self.masked:
            att = att.masked_fill(self.mask[:, :, :L, :L] == 0, float('-inf'))

        att = t_func.softmax(att, dim=-1)

        y = att @ v  # (B, nh, L, L) x (B, nh, L, hs) -> (B, nh, L, hs)
        y = y.transpose(1, 2).contiguous().view(B, L, D)  # re-assemble all head outputs side by side

        # output projection
        y = self.proj(y)
        return y


class EncodeBlock(nn.Module):
    """ an unassuming Transformer block """

    def __init__(self, embedding_size, num_heads, num_agents):
        """
        """
        super(EncodeBlock, self).__init__()

        self.ln1  = nn.LayerNorm(embedding_size)
        self.ln2  = nn.LayerNorm(embedding_size)
        self.attn = SelfAttention(embedding_size, num_heads, num_agents, masked=False)

        self.mlp  = nn.Sequential(
            init_(nn.Linear(embedding_size, 1 * embedding_size), activate=True),
            nn.GELU(),
            init_(nn.Linear(1 * embedding_size, embedding_size))
        )

    def forward(self, x):
        x = self.ln1(x + self.attn(x, x, x))
        x = self.ln2(x + self.mlp(x))
        return x


class DecodeBlock(nn.Module):
    """ an unassuming Transformer block """

    def __init__(self, embedding_size, num_heads, num_agents):
        super(DecodeBlock, self).__init__()

        self.ln1   = nn.LayerNorm(embedding_size)
        self.ln2   = nn.LayerNorm(embedding_size)
        self.ln3   = nn.LayerNorm(embedding_size)
        self.attn1 = SelfAttention(embedding_size, num_heads, num_agents, masked=True)
        self.attn2 = SelfAttention(embedding_size, num_heads, num_agents, masked=True)
        self.mlp   = nn.Sequential(
            init_(nn.Linear(embedding_size, 1 * embedding_size), activate=True),
            nn.GELU(),
            init_(nn.Linear(1 * embedding_size, embedding_size))
        )

    def forward(self, x, rep_enc):
        x = self.ln1(x + self.attn1(x, x, x))
        x = self.ln2(rep_enc + self.attn2(key=x, value=x, query=rep_enc))
        x = self.ln3(x + self.mlp(x))
        return x
#############################################################################






class MATActor(PPONetwork):

    def __init__(
        self,
        obs_space,
        action_space,
        num_agents,
        embedding_size = 64,
        num_blocks     = 1,
        num_heads      = 1,
        name           = 'actor',
        **kw_args):
        """
        """
        super(MATActor, self).__init__(
            name      = name,
            in_shape  = (embedding_size,),#FIXME: is this right?? Maybe these shapes should be None?
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

        if self.action_dtype == 'discrete':
            self.action_encoder = nn.Sequential(
                init_(nn.Linear(action_dim + 1, embedding_size, bias=False), activate=True),
                nn.GELU())
        else:
            self.action_encoder = nn.Sequential(
                init_(nn.Linear(action_dim, embedding_size), activate=True),
                nn.GELU())

        self.ln     = nn.LayerNorm(embedding_size)
        self.blocks = nn.Sequential(
            *[DecodeBlock(embedding_size, num_heads, num_agents) for _ in range(num_blocks)])

        self.head   = nn.Sequential(
            init_(nn.Linear(embedding_size, embedding_size), activate=True),
            nn.GELU(),
            nn.LayerNorm(embedding_size),
            init_(nn.Linear(embedding_size, action_dim)))

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
        embedding_size = 64,
        num_blocks     = 1,
        num_heads      = 1,
        name           = 'critic',
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
            init_(nn.Linear(self.in_size, embedding_size), activate=True),
            nn.GELU())

        self.ln     = nn.LayerNorm(embedding_size)
        self.blocks = nn.Sequential(
            *[EncodeBlock(embedding_size, num_heads, num_agents) for _ in range(num_blocks)])

        self.head   = nn.Sequential(
            init_(nn.Linear(embedding_size, embedding_size), activate=True),
            nn.GELU(),
            nn.LayerNorm(embedding_size),
            init_(nn.Linear(embedding_size, 1)))

    def encode_obs(self, obs):
        """
        """
        #print(f"ENCODING OBS OF SHAPE: {obs.shape}")#FIXME
        x = self.obs_encoder(obs)
        x = self.ln(x)
        return self.blocks(x)

    def forward(self, obs):
        # NOTE Encoder should receive a sequence of agent observations.

        # obs: (batch, num_agents, obs_dim)
        #print(f"\nEncoder obs shape: {obs.shape}")#FIXME
        # Encoder obs shape: torch.Size([N, num_agents, obs_size])

        #print(f"CRITIC GETTING OBS OF SHAPE {obs.shape}")#FIXME
        return self.head(self.encode_obs(obs))

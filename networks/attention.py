import torch
import torch.nn as nn
from torch.nn import functional as t_func
import math
import numpy as np
from ppo_and_friends.networks.utils import init_layer

from mpi4py import MPI
comm      = MPI.COMM_WORLD
rank      = comm.Get_rank()
num_procs = comm.Get_size()

class SelfAttention(nn.Module):

    def __init__(self,
                 embedding_size,
                 num_heads,
                 num_agents,
                 internal_init = 0.01,
                 out_init      = 0.01,
                 masked        = False):
        """
        NOTE: this implemenation is largey taken from the original MAT paper's
        implementation, which can be found here:
        https://github.com/PKU-MARL/Multi-Agent-Transformer

        Parameters:
        -----------
        embedding_size: int
            The size of our embedding layers.
        num_heads: int
            The number of self-attention heads to use.
        num_agents: int
            The number of agents.
        internal_init: float
            The initialization gain to use for internal layers.
        out_init: float
            The initialization gain to use for output layers.
        masked: float
            Whether or not to apply a mask that blocks out rightmost entries.
        """
        super(SelfAttention, self).__init__()

        assert embedding_size % num_heads == 0

        self.masked    = masked
        self.num_heads = num_heads

        # key, query, value projections for all heads
        self.key_net = init_layer(nn.Linear(embedding_size, embedding_size),
            gain = internal_init)
        self.query_net = init_layer(nn.Linear(embedding_size, embedding_size),
            gain = internal_init)
        self.value_net = init_layer(nn.Linear(embedding_size, embedding_size),
            gain = internal_init)

        # output projection
        self.proj = init_layer(nn.Linear(embedding_size, embedding_size),
            out_init)

        #
        # causal mask to ensure that attention is only applied to
        # the left in the input sequence
        #
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(
                num_agents + 1, num_agents + 1)).view(
                1, 1, num_agents + 1, num_agents + 1))

    def forward(self, key, value, query):
        batch_size, L, D = query.size()

        #
        # calculate query, key, values for all heads in batch and
        # move head forward to be the batch dim.
        #
        # (batch_size, num_heads, L, hs)
        k = self.key_net(key).view(
            batch_size, L, self.num_heads, D // self.num_heads).transpose(1, 2)

        # (batch_size, num_heads, L, hs)
        q = self.query_net(query).view(
            batch_size, L, self.num_heads, D // self.num_heads).transpose(1, 2)

        # (batch_size, num_heads, L, hs)
        v = self.value_net(value).view(
            batch_size, L, self.num_heads, D // self.num_heads).transpose(1, 2)

        # causal attention: (batch_size, num_heads, L, hs) x (batch_size, num_heads, hs, L)
        # -> (batch_size, num_heads, L, L)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))

        if self.masked:
            att = att.masked_fill(self.mask[:, :, :L, :L] == 0, float('-inf'))

        att = t_func.softmax(att, dim=-1)

        # (batch_size, num_heads, L, L) x (batch_size, num_heads, L, hs) -> 
        # (batch_size, num_heads, L, hs)
        y = att @ v  

        # re-assemble all head outputs side by side
        y = y.transpose(1, 2).contiguous().view(batch_size, L, D)

        # output projection
        y = self.proj(y)
        return y


class SelfAttentionEncodingBlock(nn.Module):

    def __init__(self,
                 embedding_size,
                 num_heads,
                 num_agents,
                 activation               = nn.GELU(),
                 internal_init            = nn.init.calculate_gain('relu'),
                 out_init                 = 0.01,
                 self_atten_internal_init = 0.01,
                 self_atten_out_init      = 0.01,
                 **kw_args):
        """
        NOTE: this implemenation is largey taken from the original MAT paper's
        implementation, which can be found here:
        https://github.com/PKU-MARL/Multi-Agent-Transformer

        Parameters:
        -----------
        embedding_size: int
            The size of our embedding layers.
        num_heads: int
            The number of self-attention heads to use.
        num_agents: int
            The number of agents.
        activation: function
            The activation function to use.
        internal_init: float
            The initialization gain to use for internal layers.
        out_init: float
            The initialization gain to use for output layers.
        self_atten_internal_init: float
            The initialization gain to use for internal self-attention layers.
        self_atten_out_init: float
            The initialization gain to use for output self-attention layers.
        """
        super(SelfAttentionEncodingBlock, self).__init__()

        self.ln1  = nn.LayerNorm(embedding_size)
        self.ln2  = nn.LayerNorm(embedding_size)

        self.attn = SelfAttention(
            embedding_size,
            num_heads,
            num_agents,
            internal_init = self_atten_internal_init,
            out_init      = self_atten_out_init,
            masked        = False)

        self.mlp  = nn.Sequential(
            init_layer(nn.Linear(embedding_size, 1 * embedding_size),
                gain = internal_init),

            activation,

            init_layer(nn.Linear(1 * embedding_size, embedding_size),
                gain = out_init))

    def forward(self, x):
        x = self.ln1(x + self.attn(x, x, x))
        x = self.ln2(x + self.mlp(x))
        return x


class SelfAttentionDecodingBlock(nn.Module):

    def __init__(self,
                 embedding_size,
                 num_heads,
                 num_agents,
                 mlp_hidden_scale         = 1,
                 activation               = nn.GELU(),
                 internal_init            = nn.init.calculate_gain('relu'),
                 out_init                 = 0.01,
                 self_atten_internal_init = 0.01,
                 self_atten_out_init      = 0.01,
                 **kw_args):
        """
        NOTE: this implemenation is largey taken from the original MAT paper's
        implementation, which can be found here:
        https://github.com/PKU-MARL/Multi-Agent-Transformer

        Parameters:
        -----------
        embedding_size: int
            The size of our embedding layers.
        num_heads: int
            The number of self-attention heads to use.
        num_agents: int
            The number of agents.
        mlp_hidden_scale: int
            Scale the hidden embedding layer by 
            mlp_hidden_scale * embdedding_size.
        activation: function
            The activation function to use.
        internal_init: float
            The initialization gain to use for internal layers.
        out_init: float
            The initialization gain to use for output layers.
        self_atten_internal_init: float
            The initialization gain to use for internal self-attention layers.
        self_atten_out_init: float
            The initialization gain to use for output self-attention layers.
        """

        super(SelfAttentionDecodingBlock, self).__init__()

        self.ln1   = nn.LayerNorm(embedding_size)
        self.ln2   = nn.LayerNorm(embedding_size)
        self.ln3   = nn.LayerNorm(embedding_size)

        #
        # The decoder is what we use to predict actions given previous actions,
        # so we need to mask the "future" actions in the action blocks that
        # are sent in.
        #
        self.attn1 = SelfAttention(
            embedding_size,
            num_heads,
            num_agents,
            internal_init = self_atten_internal_init,
            out_init      = self_atten_out_init,
            masked        = True)

        self.attn2 = SelfAttention(
            embedding_size,
            num_heads,
            num_agents,
            internal_init = self_atten_internal_init,
            out_init      = self_atten_out_init,
            masked        = True)

        self.mlp   = nn.Sequential(
            init_layer(
                nn.Linear(embedding_size, mlp_hidden_scale * embedding_size),
                gain = internal_init),
            activation,
            init_layer(
                nn.Linear(mlp_hidden_scale * embedding_size, embedding_size),
                gain = out_init)
        )

    def forward(self, x, rep_enc):
        x = self.ln1(x + self.attn1(x, x, x))
        x = self.ln2(rep_enc + self.attn2(key=x, value=x, query=rep_enc))
        x = self.ln3(x + self.mlp(x))
        return x

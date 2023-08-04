import torch
import torch.nn as nn
from torch.nn import functional as F
import math
import numpy as np
from torch.distributions import Categorical
from ppo_and_friends.networks.ppo_networks.base import PPONetwork

from mpi4py import MPI
comm      = MPI.COMM_WORLD
rank      = comm.Get_rank()
num_procs = comm.Get_size()

#FIXME:
#from mat.algorithms.utils.util import check, init
#from mat.algorithms.utils.transformer_act import discrete_autoregreesive_act
#from mat.algorithms.utils.transformer_act import discrete_parallel_act
#from mat.algorithms.utils.transformer_act import continuous_autoregreesive_act
#from mat.algorithms.utils.transformer_act import continuous_parallel_act

#FIXME: add optoin to our init for gain? Or, we could just keep there init function too...
def init_(m, gain=0.01, activate=False):
    if activate:
        gain = nn.init.calculate_gain('relu')
    return init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), gain=gain)


class SelfAttention(nn.Module):

    def __init__(self, embedding_size, n_head, num_agents, masked=False):
        """
        """
        super(SelfAttention, self).__init__()

        #FIXME: replace with MPI
        assert embedding_size % n_head == 0

        self.masked = masked
        self.n_head = n_head

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
        B, L, D = query.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(key).view(B, L, self.n_head, D // self.n_head).transpose(1, 2)  # (B, nh, L, hs)
        q = self.query(query).view(B, L, self.n_head, D // self.n_head).transpose(1, 2)  # (B, nh, L, hs)
        v = self.value(value).view(B, L, self.n_head, D // self.n_head).transpose(1, 2)  # (B, nh, L, hs)

        # causal attention: (B, nh, L, hs) x (B, nh, hs, L) -> (B, nh, L, L)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))

        if self.masked:
            att = att.masked_fill(self.mask[:, :, :L, :L] == 0, float('-inf'))

        att = F.softmax(att, dim=-1)

        y = att @ v  # (B, nh, L, L) x (B, nh, L, hs) -> (B, nh, L, hs)
        y = y.transpose(1, 2).contiguous().view(B, L, D)  # re-assemble all head outputs side by side

        # output projection
        y = self.proj(y)
        return y


class EncodeBlock(nn.Module):
    """ an unassuming Transformer block """

    def __init__(self, embedding_size, n_head, num_agents):
        super(EncodeBlock, self).__init__()

        self.ln1  = nn.LayerNorm(embedding_size)
        self.ln2  = nn.LayerNorm(embedding_size)
        self.attn = SelfAttention(embedding_size, n_head, num_agents, masked=False)

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

    def __init__(self, embedding_size, n_head, num_agents):
        super(DecodeBlock, self).__init__()

        self.ln1   = nn.LayerNorm(embedding_size)
        self.ln2   = nn.LayerNorm(embedding_size)
        self.ln3   = nn.LayerNorm(embedding_size)
        self.attn1 = SelfAttention(embedding_size, n_head, num_agents, masked=True)
        self.attn2 = SelfAttention(embedding_size, n_head, num_agents, masked=True)
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


class Encoder(nn.Module):

    def __init__(self, obs_dim, n_block, embedding_size, n_head, num_agents):
        super(Encoder, self).__init__()

        self.obs_dim        = obs_dim
        self.embedding_size = embedding_size
        self.num_agents     = num_agents

        self.obs_encoder = nn.Sequential(
            nn.LayerNorm(obs_dim),
            init_(nn.Linear(obs_dim, embedding_size), activate=True),
            nn.GELU())

        self.ln     = nn.LayerNorm(embedding_size)
        self.blocks = nn.Sequential(
            *[EncodeBlock(embedding_size, n_head, num_agents) for _ in range(n_block)])

        self.head   = nn.Sequential(
            init_(nn.Linear(embedding_size, embedding_size), activate=True),
            nn.GELU(),
            nn.LayerNorm(embedding_size),
            init_(nn.Linear(embedding_size, 1)))

    def forward(self, obs):
        # NOTE Encoder should receive a sequence of agent observations.

        # obs: (batch, num_agents, obs_dim)
        #print(f"\nEncoder obs shape: {obs.shape}")#FIXME
        # Encoder obs shape: torch.Size([N, 3, 21])
        obs_embeddings = self.obs_encoder(obs)
        x = obs_embeddings

        rep = self.blocks(self.ln(x))
        v_loc = self.head(rep)

        return v_loc, rep


class Decoder(nn.Module):

    def __init__(
        self,
        obs_dim,
        action_dim,
        n_block,
        embedding_size,
        n_head,
        num_agents,
        action_type='Discrete'):
        """
        """
        super(Decoder, self).__init__()

        self.action_dim = action_dim
        self.embedding_size = embedding_size
        self.action_type = action_type

        if action_type != 'Discrete':
            log_std      = torch.ones(action_dim)
            self.log_std = torch.nn.Parameter(log_std)

        else:
            if action_type == 'Discrete':
                self.action_encoder = nn.Sequential(
                    init_(nn.Linear(action_dim + 1, embedding_size, bias=False), activate=True),
                    nn.GELU())
            else:
                self.action_encoder = nn.Sequential(
                    init_(nn.Linear(action_dim, embedding_size), activate=True),
                    nn.GELU())

            self.obs_encoder = nn.Sequential(
                nn.LayerNorm(obs_dim),
                init_(nn.Linear(obs_dim, embedding_size), activate=True),
                nn.GELU())

            self.ln     = nn.LayerNorm(embedding_size)
            self.blocks = nn.Sequential(
                *[DecodeBlock(embedding_size, n_head, num_agents) for _ in range(n_block)])

            self.head   = nn.Sequential(
                init_(nn.Linear(embedding_size, embedding_size), activate=True),
                nn.GELU(),
                nn.LayerNorm(embedding_size),
                init_(nn.Linear(embedding_size, action_dim)))

    #FIXME: only used for one environment.
    #def zero_std(self, device):
    #    if self.action_type != 'Discrete':
    #        #FIXME: what's happening here?
    #        log_std = torch.zeros(self.action_dim).to(device)
    #        self.log_std.data = log_std

    def forward(self, action, obs_rep, obs):
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

        action_embeddings = self.action_encoder(action)
        x = self.ln(action_embeddings)

        for block in self.blocks:
            x = block(x, obs_rep)
        logit = self.head(x)

        return logit


class MultiAgentTransformer(PPONetwork):

    def __init__(
        self,
        obs_dim,
        action_dim,
        num_agents,
        n_block,
        embedding_size,
        n_head,
        device=torch.device("cpu"),
        action_type='Discrete'):
        """
        """
        super(MultiAgentTransformer, self).__init__()

        self.num_agents    = num_agents
        self.action_dim    = action_dim
        self.device        = device
        self.action_type   = action_type
        self.device        = device

        self.encoder = Encoder(
            obs_dim,
            n_block,
            embedding_size,
            n_head,
            num_agents)

        self.decoder = Decoder(
            obs_dim,
            action_dim,
            n_block,
            embedding_size,
            n_head,
            num_agents,
            self.action_type)
        self.to(device)

    #FIXME: only used for one env.
    #def zero_std(self):
    #    if self.action_type != 'Discrete':
    #        self.decoder.zero_std(self.device)

    def forward(self, obs, action):
        # obs: (batch, num_agents, obs_dim)
        # action: (batch, num_agents, 1)

        obs    = obs.to(self.device)
        action = action.to(self.device)

        batch_size     = obs.shape[0]
        v_loc, obs_rep = self.encoder(obs)

        if self.action_type == 'Discrete':
            action = action.long()
            action_log, entropy = discrete_parallel_act(
                self.decoder,
                obs_rep,
                obs,
                action,
                batch_size,
                self.num_agents,
                self.action_dim,
                self.device)

        else:
            action_log, entropy = continuous_parallel_act(
                self.decoder,
                obs_rep,
                obs,
                action,
                batch_size,
                self.num_agents,
                self.action_dim,
                self.device)

        return action_log, v_loc, entropy

    def get_actions(self, obs, deterministic=False):

        obs = obs.to(self.device)

        batch_size     = np.shape(obs)[0]
        v_loc, obs_rep = self.encoder(obs)

        if self.action_type == "Discrete":
            output_action, output_action_log = discrete_autoregreesive_act(
                self.decoder,
                obs_rep,
                obs,
                batch_size,
                self.num_agents,
                self.action_dim,
                self.device,
                deterministic)
        else:
            output_action, output_action_log = continuous_autoregreesive_act(
                self.decoder,
                obs_rep,
                obs,
                batch_size,
                self.num_agents,
                self.action_dim,
                self.device,
                deterministic)

        return output_action, output_action_log, v_loc

    def get_values(self, obs):
        obs = obs.to(self.device)
        v_tot, obs_rep = self.encoder(obs)
        return v_tot




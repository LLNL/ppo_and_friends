import numpy as np
import torch
from torch.optim import Adam
from ppo_and_friends.utils.episode_info import EpisodeInfo, PPODataset
from ppo_and_friends.networks.icm import ICM
from ppo_and_friends.utils.mpi_utils import rank_print
from ppo_and_friends.utils.misc import get_action_dtype
from gym.spaces import Box, Discrete, MultiDiscrete, MultiBinary
from ppo_and_friends.utils.mpi_utils import broadcast_model_parameters

from mpi4py import MPI
comm      = MPI.COMM_WORLD
rank      = comm.Get_rank()
num_procs = comm.Get_size()

class AgentPolicy():

    def __init__(self,
                 action_space,
                 actor_observation_space,
                 critic_observation_space,
                 ac_network,
                 icm_network,
                 actor_kw_args,
                 critic_kw_args,
                 icm_kw_args,
                 lr,
                 device,
                 enable_icm  = False,
                 test_mode   = False):
        """
        """
        self.action_space     = action_space
        self.actor_obs_space  = actor_observation_space
        self.critic_obs_space = critic_observation_space
        self.enable_icm       = enable_icm
        self.device           = device
        self.test_mode        = test_mode

        act_type = type(action_space)

        if (issubclass(act_type, Box) or
            issubclass(act_type, MultiBinary) or
            issubclass(act_type, MultiDiscrete)):

            self.act_dim = action_space.shape

        elif issubclass(act_type, Discrete):
            self.act_dim = action_space.n

        else:
            msg = "ERROR: unsupported action space {}".format(action_space)
            rank_print(msg)
            comm.Abort()

        if ((issubclass(act_type, MultiBinary) or
             issubclass(act_type, MultiDiscrete)) and
             (not is_multi_agent)):
            msg  = "WARNING: MultiBinary and MultiDiscrete action spaces "
            msg += "may not be fully supported. Use at your own risk."
            rank_print(msg)

        self.action_dtype = get_action_dtype(self.action_space)

        if self.action_dtype == "unknown":
            rank_print("ERROR: unknown action type!")
            comm.Abort()
        else:
            rank_print("Using {} actions.".format(self.action_dtype))


        self._initialize_networks(
            ac_network     = ac_network,
            enable_icm     = enable_icm,
            icm_network    = icm_network,
            actor_kw_args  = actor_kw_args,
            critic_kw_args = critic_kw_args,
            icm_kw_args    = icm_kw_args)

        self.actor_optim  = Adam(self.actor.parameters(), lr=lr, eps=1e-5)
        self.critic_optim = Adam(self.critic.parameters(), lr=lr, eps=1e-5)

        if self.enable_icm:
            self.icm_optim = Adam(self.icm_model.parameters(),
                lr=lr, eps=1e-5)

    def _initialize_networks(self,
                             ac_network, 
                             enable_icm,
                             icm_network,
                             actor_kw_args,
                             critic_kw_args,
                             icm_kw_args):
        """
        """
        #
        # Initialize our networks: actor, critic, and possibly ICM.
        #
        use_conv2d_setup = False
        for base in ac_network.__bases__:
            if base.__name__ == "PPOConv2dNetwork":
                use_conv2d_setup = True

        self.using_lstm = False
        for base in ac_network.__bases__:
            if base.__name__ == "PPOLSTMNetwork":
                self.using_lstm = True

        #
        # arXiv:2006.05990v1 suggests initializing the output layer
        # of the actor network with a weight that's ~100x smaller
        # than the rest of the layers. We initialize layers with a
        # value near 1.0 by default, so we set the last layer to
        # 0.01. The same paper also suggests that the last layer of
        # the value network doesn't matter so much. I can't remember
        # where I got 1.0 from... I'll try to track that down.
        #
        if use_conv2d_setup:
            obs_dim = self.actor_obs_space.shape

            self.actor = ac_network(
                name         = "actor", 
                in_shape     = obs_dim,
                out_dim      = self.act_dim, 
                out_init     = 0.01,
                action_dtype = self.action_dtype,
                test_mode    = self.test_mode,
                **actor_kw_args)

            self.critic = ac_network(
                name         = "critic", 
                in_shape     = obs_dim,
                out_dim      = 1,
                out_init     = 1.0,
                action_dtype = self.action_dtype,
                test_mode    = self.test_mode,
                **critic_kw_args)

        else:
            actor_obs_dim  = self.actor_obs_space.shape[0]
            critic_obs_dim = self.critic_obs_space.shape[0]

            self.actor = ac_network(
                name         = "actor", 
                in_dim       = actor_obs_dim,
                out_dim      = self.act_dim, 
                out_init     = 0.01,
                action_dtype = self.action_dtype,
                test_mode    = self.test_mode,
                **actor_kw_args)

            self.critic = ac_network(
                name         = "critic", 
                in_dim       = critic_obs_dim,
                out_dim      = 1,
                out_init     = 1.0,
                action_dtype = self.action_dtype,
                test_mode    = self.test_mode,
                **critic_kw_args)

        self.actor  = self.actor.to(self.device)
        self.critic = self.critic.to(self.device)

        broadcast_model_parameters(self.actor)
        broadcast_model_parameters(self.critic)
        comm.barrier()

        if enable_icm:
            obs_dim = self.actor_obs_shape[0]
            self.icm_model = icm_network(
                name         = "icm",
                obs_dim      = obs_dim,
                act_dim      = self.act_dim,
                action_dtype = self.action_dtype,
                test_mode    = self.test_mode,
                **icm_kw_args)

            self.icm_model.to(self.device)
            broadcast_model_parameters(self.icm_model)
            comm.barrier()

    def get_action(self, obs):
        """
            Given an observation from our environment, determine what the
            action should be.

            Arguments:
                obs    The environment observation.

            Returns:
                A tuple of form (raw_action, action, log_prob) s.t. "raw_action"
                is the distribution sample before any "squashing" takes place,
                "action" is the the action value that should be fed to the
                environment, and log_prob is the log probabilities from our
                probability distribution.
        """
        if len(obs.shape) < 2:
            msg  = "ERROR: get_action expects a batch of observations but "
            msg += "instead received shape {}.".format(obs.shape)
            rank_print(msg)
            comm.Abort()

        t_obs = torch.tensor(obs, dtype=torch.float).to(self.device)

        with torch.no_grad():
            action_pred = self.actor(t_obs)

        action_pred = action_pred.cpu().detach()
        dist        = self.actor.distribution.get_distribution(action_pred)

        #
        # Our distribution gives us two potentially distinct actions, one of
        # which is guaranteed to be a raw sample from the distribution. The
        # other might be altered in some way (usually to enforce a range).
        #
        action, raw_action = self.actor.distribution.sample_distribution(dist)
        log_prob = self.actor.distribution.get_log_probs(dist, raw_action)

        action     = action.detach().numpy()
        raw_action = raw_action.detach().numpy()

        return raw_action, action, log_prob.detach()

    def evaluate(self, batch_critic_obs, batch_obs, batch_actions):
        """
            Given a batch of observations, use our critic to approximate
            the expected return values. Also use a batch of corresponding
            actions to retrieve some other useful information.

            Arguments:
                batch_critic_obs   A batch of observations for the critic.
                batch_obs          A batch of standard observations.
                batch_actions      A batch of actions corresponding to the batch of
                                   observations.

            Returns:
                A tuple of form (values, log_probs, entropies) s.t. values are
                the critic predicted value, log_probs are the log probabilities
                from our probability distribution, and entropies are the
                entropies from our distribution.
        """
        values      = self.critic(batch_critic_obs).squeeze()
        action_pred = self.actor(batch_obs).cpu()
        dist        = self.actor.distribution.get_distribution(action_pred)

        if self.action_dtype == "continuous" and len(batch_actions.shape) < 2:
            log_probs = self.actor.distribution.get_log_probs(
                dist,
                batch_actions.unsqueeze(1).cpu())
        else:
            log_probs = self.actor.distribution.get_log_probs(
                dist,
                batch_actions.cpu())

        entropy = self.actor.distribution.get_entropy(dist, action_pred)

        return values, log_probs.to(self.device), entropy.to(self.device)

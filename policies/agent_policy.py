import numpy as np
import torch
from torch.optim import Adam
from ppo_and_friends.utils.episode_info import EpisodeInfo, PPODataset
from ppo_and_friends.networks.icm import ICM
from ppo_and_friends.utils.mpi_utils import rank_print
from ppo_and_friends.utils.misc import get_action_dtype
from gym.spaces import Box, Discrete, MultiDiscrete, MultiBinary
from ppo_and_friends.utils.mpi_utils import broadcast_model_parameters
from ppo_and_friends.utils.misc import update_optimizer_lr

from mpi4py import MPI
comm      = MPI.COMM_WORLD
rank      = comm.Get_rank()
num_procs = comm.Get_size()

# FIXME: do we want to have two different modes, one for single agent
# and another for multi-agent?
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
                 use_gae,
                 gamma,
                 lambd,
                 dynamic_bs_clip,
                 bootstrap_clip,
                 status_dict,
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
        self.use_gae          = use_gae
        self.gamma            = gamma
        self.lambd            = lambd
        self.dynamic_bs_clip  = dynamic_bs_clip
        self.bootstrap_clip   = bootstrap_clip
        self.status_dict      = status_dict
        self.using_lstm       = False
        self.dataset          = None
        self.episodes         = None

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

    def initialize_episodes(self, env_batch_size):
        """
        """
        #
        # FIXME: when tracking episode infos in the policy (or wherever)
        # we need to map TWO things to EpisodeInfo objects:
        #    1. Agents
        #    2. Environment instances!
        #
        # The policy could keep arrays of dictionaries s.t. each dict
        # in the array is an env instance, and each dict maps agents
        # to episodes.
        #
        # NOTE that different agents will map to different polices, meaning
        # that our dictionaries can be different sizes for each policy, but
        # the number of environment instances will be consistent.
        #
        self.episodes = np.array([None] * env_batch_size, dtype=object)
        bs_clip_range = self.get_bs_clip_range(None)

        # FIXME: these will become dictionaries mapping agent ids to
        # their episodes.
        for ei_idx in range(env_batch_size):
            self.episodes[ei_idx] = EpisodeInfo(
                starting_ts    = 0,
                use_gae        = self.use_gae,
                gamma          = self.gamma,
                lambd          = self.lambd,
                bootstrap_clip = bs_clip_range)

    def initialize_dataset(self):
        """
        """
        sequence_length = 1
        if self.using_lstm:
            self.actor.reset_hidden_state(
                batch_size = 1,
                device     = self.device)

            self.critic.reset_hidden_state(
                batch_size = 1,
                device     = self.device)

            sequence_length = self.actor.sequence_length

        self.dataset = PPODataset(
            device          = self.device,
            action_dtype    = self.action_dtype,
            sequence_length = sequence_length)

    def add_episode_info(
        self, 
        global_observations,
        observations,
        next_observations,
        raw_actions, 
        actions, 
        values, 
        log_probs, 
        rewards, 
        actor_hidden, 
        actor_cell, 
        critic_hidden, 
        critic_cell):
        """
        """
        # FIXME: these will become dictionaries mapping agent ids
        # to their episode infos.
        env_batch_size = self.episodes.size

        for ei_idx in range(env_batch_size):
            self.episodes[ei_idx].add_info(
                global_observation = global_observations[ei_idx],
                observation        = observations[ei_idx],
                next_observation   = next_observations[ei_idx],
                raw_action         = raw_actions[ei_idx],
                action             = actions[ei_idx],
                value              = values[ei_idx].item(),
                log_prob           = log_probs[ei_idx],
                reward             = rewards[ei_idx].item(),
                actor_hidden       = actor_hidden[:, [ei_idx], :],
                actor_cell         = actor_cell[:, [ei_idx], :],
                critic_hidden      = critic_hidden[:, [ei_idx], :],
                critic_cell        = critic_cell[:, [ei_idx], :])

    def end_episodes(
        self,
        env_idxs,
        episode_lengths,
        terminal,
        ending_values,
        ending_rewards):
        """
        """
        for idx, env_i in enumerate(env_idxs):
            self.episodes[env_i].end_episode(
                ending_ts      = episode_lengths[env_i],
                terminal       = terminal[idx],
                ending_value   = ending_values[idx].item(),
                ending_reward  = ending_rewards[idx].item())

            self.dataset.add_episode(self.episodes[env_i])

        #
        # If we're using a dynamic bs clip, we clip to the min/max
        # rewards from the episode. Otherwise, rely on the user
        # provided range.
        #
        for idx, env_i in enumerate(env_idxs):
            bs_min, bs_max = self.get_bs_clip_range(
                self.episodes[env_i].rewards)

            #
            # If we're terminal, the start of the next episode is 0.
            # Otherwise, we pick up where we left off.
            #
            starting_ts = 0 if terminal[env_i] else episode_lengths[env_i]

            self.episodes[env_i] = EpisodeInfo(
                starting_ts    = starting_ts,
                use_gae        = self.use_gae,
                gamma          = self.gamma,
                lambd          = self.lambd,
                bootstrap_clip = (bs_min, bs_max))

    def finalize_dataset(self):
        """
        """
        self.dataset.build()

    def clear_dataset(self):
        """
        """
        self.dataset = None

    #FIXME: obs will be a dictionary in the multi-agent case. Do we want this to
    # be handed by a multi-agent mode, or do we want to wrap single agent
    # envs so that they also return dictionaries? OR we could just make sure to
    # only pass observations from the correct agents into this function... Maybe
    # that makes the most sense.
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

    def get_intrinsic_reward(self,
                             prev_obs,
                             obs,
                             action):
        """
            Query the ICM for an intrinsic reward.

            Arguments:
                prev_obs    The previous observation (before the latest
                            action).
                obs         The current observation.
                action      The action taken.
        """
        if len(obs.shape) < 2:
            msg  = "ERROR: get_intrinsic_reward expects a batch of "
            msg += "observations but "
            msg += "instead received shape {}.".format(obs.shape)
            rank_print(msg)
            comm.Abort()

        obs_1 = torch.tensor(prev_obs,
            dtype=torch.float).to(self.device)
        obs_2 = torch.tensor(obs,
            dtype=torch.float).to(self.device)

        if self.action_dtype == "discrete":
            action = torch.tensor(action,
                dtype=torch.long).to(self.device)

        elif self.action_dtype == "continuous":
            action = torch.tensor(action,
                dtype=torch.float).to(self.device)

        if len(action.shape) != 2:
            action = action.unsqueeze(1)

        with torch.no_grad():
            intr_reward, _, _ = self.icm_model(obs_1, obs_2, action)

        batch_size   = obs.shape[0]
        intr_reward  = intr_reward.detach().cpu().numpy()
        intr_reward  = intr_reward.reshape((batch_size, -1))
        intr_reward *= self.intr_reward_weight

        return intr_reward

    def update_learning_rate(self, lr):
        """
        """
        update_optimizer_lr(self.actor_optim, lr)
        update_optimizer_lr(self.critic_optim, lr)

        if self.enable_icm:
            update_optimizer_lr(self.icm_optim, lr)

    def get_bs_clip_range(self, ep_rewards):
        """
            Get the current bootstrap clip range.

            Arguments:
                ep_rewards    A numpy array containing the rewards for
                              this episode.

            Returns:
                A tuple containing the min and max values for the bootstrap
                clip.
        """
        if self.dynamic_bs_clip and ep_rewards is not None:
            bs_min = min(ep_rewards)
            bs_max = max(ep_rewards)

        else:
            iteration = self.status_dict["iteration"]
            timestep  = self.status_dict["timesteps"]

            bs_min = self.bootstrap_clip[0](
                iteration = iteration,
                timestep  = timestep)

            bs_max = self.bootstrap_clip[1](
                iteration = iteration,
                timestep  = timestep)

        return (bs_min, bs_max)

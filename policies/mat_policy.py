import numpy as np
import os
import torch
from torch.nn import functional as t_func
from copy import deepcopy
from functools import reduce
from torch.optim import Adam
from ppo_and_friends.utils.episode_info import EpisodeInfo, PPODataset, PPOSharedEpisodeDataset
from ppo_and_friends.networks.ppo_networks.icm import ICM
from ppo_and_friends.utils.mpi_utils import rank_print
from ppo_and_friends.utils.misc import get_action_dtype
from gymnasium.spaces import Box, Discrete, MultiDiscrete, MultiBinary
from ppo_and_friends.utils.mpi_utils import broadcast_model_parameters
from ppo_and_friends.utils.misc import update_optimizer_lr
from ppo_and_friends.networks.ppo_networks.feed_forward import FeedForwardNetwork
from ppo_and_friends.networks.actor_critic.wrappers import to_actor, to_critic
import ppo_and_friends.networks.actor_critic.multi_agent_transformer as mat
from ppo_and_friends.utils.schedulers import LinearScheduler, CallableValue
from ppo_and_friends.utils.misc import get_flattened_space_length
from ppo_and_friends.policies.agent_policy import AgentPolicy

from mpi4py import MPI
comm      = MPI.COMM_WORLD
rank      = comm.Get_rank()
num_procs = comm.Get_size()


class MATPolicy(AgentPolicy):
    """
    """

    def __init__(self, *args, **kw_args):
        """
        """
        super(MATPolicy, self).__init__(*args, **kw_args)
        self.action_dim     = get_flattened_space_length(self.action_space)
        self.agent_grouping = True

    #TODO: update to use MAT
    def _initialize_networks(
        self,
        enable_icm,
        icm_network,
        actor_kw_args,
        critic_kw_args,
        icm_kw_args,
        actor_network  = mat.MATActor,
        critic_network = mat.MATCritic):
        """
        Initialize our networks.

        Parameters:
        -----------
        enable_icm: bool
            Whether or not to enable ICM.
        icm_network: class of type PPONetwork
            The network class to use for ICM (when enabled).
        actor_kw_args: dict
            Keyword args for the actor network.
        critic_kw_args: dict
            Keyword args for the critic network.
        icm_kw_args: dict
            Keyword args for the ICM network.
        actor_network: class of type PPONetwork
            The network to use for the actor.
        critic_network: class of type PPONetwork
            The network to use for the critic.
        """
        #
        # Initialize our networks: actor, critic, and possibly ICM.
        #
        # FIXME: let's try integrating this into the MAT networks.
        #
        # arXiv:2006.05990v1 suggests initializing the output layer
        # of the actor network with a weight that's ~100x smaller
        # than the rest of the layers. We initialize layers with a
        # value near 1.0 by default, so we set the last layer to
        # 0.01. The same paper also suggests that the last layer of
        # the value network doesn't matter so much. I can't remember
        # where I got 1.0 from... I'll try to track that down.
        #


        self.actor = actor_network(
            name         = "actor", 
            obs_space    = self.actor_obs_space,
            #out_init     = 0.01,#FIXME
            action_space = self.action_space,
            num_agents   = len(self.agent_ids),
            test_mode    = self.test_mode,
            **actor_kw_args)

        self.critic = critic_network(
            name         = "critic", 
            obs_space    = self.critic_obs_space,
            #out_init     = 1.0,#FIXME
            num_agents   = len(self.agent_ids),
            test_mode    = self.test_mode,
            **critic_kw_args)

        self.actor  = self.actor.to(self.device)
        self.critic = self.critic.to(self.device)

        broadcast_model_parameters(self.actor)
        broadcast_model_parameters(self.critic)
        comm.barrier()

        if enable_icm:
            self.icm_model = icm_network(
                name         = "icm",
                obs_space    = self.actor_obs_space,
                action_space = self.action_space,
                test_mode    = self.test_mode,
                **icm_kw_args)

            self.icm_model = self.icm_model.to(self.device)
            broadcast_model_parameters(self.icm_model)
            comm.barrier()

    def initialize_dataset(self):
        """
            Initialize a rollout dataset. This should be called at the
            onset of a rollout.
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

        self.dataset = PPOSharedEpisodeDataset(
            device          = self.device,
            action_dtype    = self.action_dtype,
            sequence_length = sequence_length,
            num_envs        = self.envs_per_proc,
            agent_ids       = self.agent_ids)

    # TODO: this is identical to the general case excpet for how
    # we add data to our dataset. There may be a better approach to this...
    def end_episodes(
        self,
        agent_id,
        env_idxs,
        episode_lengths,
        terminal,
        ending_values,
        ending_rewards):
        """
            End a rollout episode.

            Arguments:
                agent_id         The associated agent id.
                env_idxs         The associated environment indices.
                episode_lengths  The lenghts of the ending episode(s).
                terminal         Which episodes are terminally ending.
                ending_values    Ending values for the episode(s).
                ending_rewards   Ending rewards for the episode(s)
        """
        self.validate_agent_id(agent_id)

        for idx, env_i in enumerate(env_idxs):
            self.episodes[agent_id][env_i].end_episode(
                ending_ts      = episode_lengths[env_i],
                terminal       = terminal[idx],
                ending_value   = ending_values[idx].item(),
                ending_reward  = ending_rewards[idx].item())

            self.dataset.add_shared_episode(
                self.episodes[agent_id][env_i],
                agent_id,
                env_i)

            #
            # If we're using a dynamic bs clip, we clip to the min/max
            # rewards from the episode. Otherwise, rely on the user
            # provided range.
            #
            bs_min, bs_max = self.get_bs_clip_range(
                self.episodes[agent_id][env_i].rewards)

            #
            # If we're terminal, the start of the next episode is 0.
            # Otherwise, we pick up where we left off.
            #
            starting_ts = 0 if terminal[idx] else episode_lengths[env_i]

            self.episodes[agent_id][env_i] = EpisodeInfo(
                starting_ts    = starting_ts,
                use_gae        = self.use_gae,
                gamma          = self.gamma,
                lambd          = self.lambd,
                bootstrap_clip = (bs_min, bs_max))

    def _get_tokened_action_block(self, batch_size):
        """
        """
        num_agents = len(self.agent_ids),

        if self.action_dtype == "continuous":
            action_block = torch.zeros(
               (batch_size, num_agents, self.action_dim)).to(device)

        elif self.action_dtype == "discrete":
            action_block = torch.zeros(
                (batch_size, num_agents, self.action_dim + 1)).to(self.device)
            action_block[:, 0, 0] = 1

        return action_block

    def _get_parallel_actions(encoded_obs):
        """
        """
        encoded_obs = torch.tensor(encoded_obs, dtype=torch.float)
        encoded_obs = encoded_obs.to(self.device)
    
        # FIXME: This might break during rollouts becauset the observations
        # might be of shape (obs_size,).
        batch_size = encoded_obs.shape[0]
        num_agents = len(self.agent_ids),

        action_block = self._get_tokened_action_block(batch_size)

        #FIXME: is there a better way to handle this?
        action_offset = 0
        if self.action_dtype == "discrete":
            action_offset = 1

        action_block[:, 1:, action_offset:] = action[:, :-1, :]
    
        with torch.no_grad():
            action_pred = self.actor(action_block, encoded_obs)

        dist = self.actor.distribution.get_distribution(action_pred)
        action, raw_action = self.actor.distribution.sample_distribution(dist)
        log_prob = self.actor.distribution.get_log_probs(dist, raw_action)

        return action, raw_action, log_prob

    def _get_autoregressive_actions(encoded_obs):
        """
        """
        # FIXME: This might break during rollouts becauset the observations
        # might be of shape (obs_size,).
        batch_size = encoded_obs.shape[0]
        num_agents = len(self.agent_ids),

        action_block = self._get_tokened_action_block(batch_size)

        output_action     = torch.zeros((batch_size, num_agents, 1), dtype=torch.long)#FIXME: change this to float32
        output_raw_action = torch.zeros_like(output_action, dtype=torch.float32)
        output_action_log = torch.zeros_like(output_action, dtype=torch.float32)

        #FIXME: is there a better way to handle this?
        action_offset = 0
        if self.action_dtype == "discrete":
            action_offset = 1

        with torch.no_grad():
            for i in range(num_agents):
                action_pred = self.actor(action_block, encoded_obs)[:, i, :]
                dist        = self.actor.distribution.get_distribution(action_pred)

                action, raw_action = self.actor.distribution.sample_distribution(dist)
                log_prob = self.actor.distribution.get_log_probs(dist, raw_action)

                #FIXME: is there a better way of doing this?
                if self.action_dtype == "discrete":
                    output_action[:, i, :]     = action.unsqueeze(-1)
                    output_raw_action[:, i, :] = raw_action.unsqueeze(-1)
                    output_action_log[:, i, :] = action_log.unsqueeze(-1)
                    action = t_func.one_hot(action, num_classes=self.action_dim)
                else:
                    output_action[:, i, :]     = action
                    output_raw_action[:, i, :] = raw_action
                    output_action_log[:, i, :] = action_log

                if i + 1 < num_agents:
                    action_block[:, i + 1, action_offset:] = action

        return output_action, output_raw_action, output_action_log

    def get_rollout_actions(self, obs):
        """

        Returns:
        --------
        tuple
            A tuple of form (raw_action, action, log_prob) s.t. "raw_action"
            is the distribution sample before any "squashing" takes place,
            "action" is the the action value that should be fed to the
            environment, and log_prob is the log probabilities from our
            probability distribution.
        """
        if len(obs.shape) < 2:
            msg  = "ERROR: _get_action_with_exploration expects a "
            msg ++ "batch of observations but "
            msg += "instead received shape {}.".format(obs.shape)
            rank_print(msg)
            comm.Abort()

        # FIXME: during rollouts, our agents are always acting in the same order.
        # we can shuffle them between rollouts, but I think that's all we have
        # right now. Is that okay???

        #
        # The incoming shape is (num_agents, num_envs, obs_size). MAT wants
        # (num_envs, num_agents, obs_size).
        #
        obs = np.swapaxes(obs, 0, 1)
        t_obs = torch.tensor(obs, dtype=torch.float)
        t_obs = t_obs.to(self.device)

        encoded_obs = self.critic.encode_obs(t_obs)
        actions, raw_actions, log_probs = self._get_autoregressive_actions(encoded_obs)

        actions     = torch.swapaxes(actions, 0, 1)
        raw_actions = torch.swapaxes(raw_actions, 0, 1)
        log_probs   = torch.swapaxes(log_probs, 0, 1)

        # FIXME: this is reverse order from the distributions, 
        # which is a bit confusing. It's all over the place...
        return raw_actions, actions, log_probs.detach()

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
        #
        # The incoming shape is (num_agents, num_envs, obs_size). MAT wants
        # (num_envs, num_agents, obs_size).
        #
        batch_critic_obs = torch.swapaxes(batch_critic_obs, 0, 1)
        batch_obs        = torch.swapaxes(batch_obs, 0, 1)
        batch_actions    = torch.swapaxes(batch_actions, 0, 1)

        values      = self.critic(batch_critic_obs).squeeze()#FIXME: do we want this squeeze?
        encoded_obs = self.critic.encode_obs(batch_critic_obs)
        action_pred = self._get_parallel_actions(encoded_obs)
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

        values    = torch.swapaxes(values, 0, 1)
        log_probs = torch.swapaxes(log_probs, 0, 1)
        entropy   = torch.swapaxes(entropy, 0, 1)

        return values, log_probs.to(self.device), entropy.to(self.device)

    def get_inference_actions(self, obs, explore):
        """
        Given observations from our environment, determine what the
        actions should be.

        This method is meant to be used for inference only, and it
        will return the environment actions alone.

        Arguments:
            obs       The environment observation.
            explore   Should we allow exploration?

        Returns:
            Predicted actions to perform in the environment.
        """
        # FIXME: we need to combine and the "un-combine" agents.
        # we should be able to use the "get_policy_batches" function
        # in ppo.py. Let's turn that into a utility outside of that class.
        if explore:
            return self._get_action_with_exploration(obs)
        return self._get_action_without_exploration(obs)

    def _get_actions_with_exploration(self, obs):
        """
        Given observations from our environment, determine what the
        next actions should be taken while allowing natural exploration.

        Arguments:
            obs    The environment observations.

        Returns:
            A tuple of form (raw_action, action, log_prob) s.t. "raw_action"
            is the distribution sample before any "squashing" takes place,
            "action" is the the action value that should be fed to the
            environment, and log_prob is the log probabilities from our
            probability distribution.
        """
        if len(obs.shape) < 3:
            msg  = "ERROR: _get_action_with_exploration expects a "
            msg ++ "batch of agent observations but "
            msg += "instead received shape {}.".format(obs.shape)
            rank_print(msg)
            comm.Abort()

        #
        # The incoming shape is (num_agents, num_envs, obs_size). MAT wants
        # (num_envs, num_agents, obs_size).
        #
        obs = np.swapaxes(obs, 0, 1)
        t_obs = torch.tensor(obs, dtype=torch.float)
        t_obs = t_obs.to(self.device)

        encoded_obs = self.critic.encode_obs(t_obs)
        actions, _, _ = self._get_autoregressive_actions(encoded_obs)

        actions = torch.swapaxes(actions, 0, 1)

        return actions

    #TODO: upate for MAT
    def _get_actions_without_exploration(self, obs):
        """
        Given observations from our environment, determine what the
        next actions should be while not allowing any exploration.

        Arguments:
            obs    The environment observations.

        Returns:
            The next actions to perform.
        """
        if len(obs.shape) < 3:
            msg  = "ERROR: _get_action_without_exploration expects a "
            msg ++ "batch of agent observations but "
            msg += "instead received shape {}.".format(obs.shape)
            rank_print(msg)
            comm.Abort()

        #t_obs = torch.tensor(obs, dtype=torch.float).to(self.device)

        #with torch.no_grad():
        #    return self.actor.get_refined_prediction(t_obs)

        #
        # The incoming shape is (num_agents, num_envs, obs_size). MAT wants
        # (num_envs, num_agents, obs_size).
        #
        obs = np.swapaxes(obs, 0, 1)
        t_obs = torch.tensor(obs, dtype=torch.float)
        t_obs = t_obs.to(self.device)

        encoded_obs = self.critic.encode_obs(t_obs)

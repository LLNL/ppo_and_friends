import numpy as np
import os
import torch
from functools import reduce
from torch.optim import Adam
from ppo_and_friends.utils.episode_info import EpisodeInfo, PPODataset
from ppo_and_friends.networks.icm import ICM
from ppo_and_friends.utils.mpi_utils import rank_print
from ppo_and_friends.utils.misc import get_action_dtype
from gym.spaces import Box, Discrete, MultiDiscrete, MultiBinary
from ppo_and_friends.utils.mpi_utils import broadcast_model_parameters
from ppo_and_friends.utils.misc import update_optimizer_lr
from ppo_and_friends.networks.actor_critic_networks import FeedForwardNetwork

from mpi4py import MPI
comm      = MPI.COMM_WORLD
rank      = comm.Get_rank()
num_procs = comm.Get_size()

class AgentPolicy():

    def __init__(self,
                 name,
                 action_space,
                 actor_observation_space,
                 critic_observation_space,
                 bootstrap_clip     = (-10., 10.),
                 ac_network         = FeedForwardNetwork,
                 actor_kw_args      = {},
                 critic_kw_args     = {},
                 icm_kw_args        = {},
                 lr                 = 3e-4,
                 use_gae            = True,
                 gamma              = 0.99,
                 lambd              = 0.95,
                 dynamic_bs_clip    = False,
                 enable_icm         = False,
                 icm_network        = ICM,
                 intr_reward_weight = 1.0,
                 test_mode          = False):
        """
            Arguments:
                 lambd                The 'lambda' value for calculating GAEs.
                 use_gae              Should we use Generalized Advantage
                                      Estimations? If not, fall back on the
                                      vanilla advantage calculation.
                 bootstrap_clip       When using GAE, we bootstrap the values
                                      and rewards when an epsiode is cut off
                                      before completion. In these cases, we
                                      clip the bootstrapped reward to a
                                      specific range. Why is this? Well, our
                                      estimated reward (from our value network)
                                      might be way outside of the expected
                                      range. We also allow the range min/max
                                      to be callables that take in the
                                      current iteration.
                 dynamic_bs_clip      If set to True, bootstrap_clip will be
                                      used as the initial clip values, but all
                                      values thereafter will be taken from the
                                      global min and max rewards that have been
                                      seen so far.

        """
        self.name               = name
        self.action_space       = action_space
        self.actor_obs_space    = actor_observation_space
        self.critic_obs_space   = critic_observation_space
        self.enable_icm         = enable_icm
        self.intr_reward_weight = intr_reward_weight
        self.test_mode          = test_mode
        self.use_gae            = use_gae
        self.gamma              = gamma
        self.lambd              = lambd
        self.dynamic_bs_clip    = dynamic_bs_clip
        self.using_lstm         = False
        self.dataset            = None
        self.device             = torch.device("cpu")
        self.agent_ids          = set()
        self.episodes           = {}

        act_type = type(action_space)
        self.act_nvec = None

        if issubclass(act_type, Box):
            self.act_dim = action_space.shape

        elif (issubclass(act_type, Discrete) or
            issubclass(act_type, MultiBinary)):
            self.act_dim = action_space.n

        elif issubclass(act_type, MultiDiscrete):
            self.act_dim  = reduce(lambda a, b: a+b, action_space.nvec)
            self.act_nvec = action_space.nvec

        else:
            msg = "ERROR: unsupported action space {}".format(action_space)
            rank_print(msg)
            comm.Abort()

        self.action_dtype = get_action_dtype(self.action_space)

        if self.action_dtype == "unknown":
            msg  = "ERROR: unknown action type: "
            msg += f"{type(self.action_space)} with dtype "
            msg += f"{self.action_space.dtype}."
            rank_print(msg)
            comm.Abort()
        else:
            rank_print("Using {} actions.".format(self.action_dtype))

        #
        # One or both of our bootstrap clip ends might be a function of
        # our iteration.
        # We turn them both into functions for sanity.
        #
        min_bs_callable  = None
        max_bs_callable  = None
        bs_clip_callable = False

        if callable(bootstrap_clip[0]):
            min_bs_callable  = bootstrap_clip[0]
            bs_clip_callable = True
        else:
            min_bs_callable = lambda *args, **kwargs : bootstrap_clip[0]

        if callable(bootstrap_clip[1]):
            max_bs_callable  = bootstrap_clip[1]
            bs_clip_callable = True
        else:
            max_bs_callable = lambda *args, **kwargs : bootstrap_clip[1]

        self.bootstrap_clip = (min_bs_callable, max_bs_callable)

        if bs_clip_callable and dynamic_bs_clip:
            msg  = "WARNING: it looks like you've enabled dynamic_bs_clip "
            msg += "and also set the bootstrap clip to be callables. This is "
            msg += "redundant, and the dynamic clip will override the given "
            msg += "functions."
            rank_print(msg)

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
        else:
            self.icm_optim = None

    def register_agent(self, agent_id):
        """
        """
        self.agent_ids = self.agent_ids.union({agent_id})

    def to(self, device):
        """
        """
        self.device    = device
        self.actor     = self.actor.to(self.device)
        self.critic    = self.critic.to(self.device)

        if self.enable_icm:
            self.icm_model = self.icm_model.to(self.device)

    def _initialize_networks(
        self,
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
            self.actor = ac_network(
                name         = "actor", 
                in_shape     = self.actor_obs_space.shape,
                out_dim      = self.act_dim, 
                out_init     = 0.01,
                action_nvec  = self.act_nvec,
                action_dtype = self.action_dtype,
                test_mode    = self.test_mode,
                **actor_kw_args)

            self.critic = ac_network(
                name         = "critic", 
                in_shape     = self.critic_obs_space.shape,
                out_dim      = 1,
                out_init     = 1.0,
                action_dtype = self.action_dtype,
                test_mode    = self.test_mode,
                **critic_kw_args)

        else:
            if type(self.actor_obs_space) == Box:
                actor_obs_dim  = self.actor_obs_space.shape[0]
                critic_obs_dim = self.critic_obs_space.shape[0]

            elif type(self.actor_obs_space) == Discrete:
                actor_obs_dim  = self.actor_obs_space.n
                critic_obs_dim = self.critic_obs_space.n

            else:
                msg  = f"ERROR: {type(self.actor_obs_space)} is not a "
                msg += "supported observation space."
                rank_print(msg)
                comm.Abort()

            self.actor = ac_network(
                name         = "actor", 
                in_dim       = actor_obs_dim,
                out_dim      = self.act_dim, 
                out_init     = 0.01,
                action_nvec  = self.act_nvec,
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
            obs_dim = self.actor_obs_space.shape[0]

            self.icm_model = icm_network(
                name         = "icm",
                obs_dim      = obs_dim,
                act_dim      = self.act_dim,
                action_nvec  = self.act_nvec,
                action_dtype = self.action_dtype,
                test_mode    = self.test_mode,
                **icm_kw_args)

            self.icm_model = self.icm_model.to(self.device)
            broadcast_model_parameters(self.icm_model)
            comm.barrier()

    def initialize_episodes(self, env_batch_size, status_dict):
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

        self.episodes = {}
        for agent_id in self.agent_ids:
            self.episodes[agent_id] = np.array([None] * env_batch_size,
                dtype=object)

            bs_clip_range = self.get_bs_clip_range(None, status_dict)

            for ei_idx in range(env_batch_size):
                self.episodes[agent_id][ei_idx] = EpisodeInfo(
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

    def validate_agent_id(self, agent_id):
        """
        """
        if agent_id not in self.agent_ids:
            msg  = f"ERROR: agent {agent_id} has not been registered with "
            msg += "policy {self.name}. Make sure that you've set up your "
            msg += "policies correctly."
            rank_print(msg)
            comm.Abort()

    def add_episode_info(
        self, 
        agent_id,
        global_observations,
        observations,
        next_observations,
        raw_actions, 
        actions, 
        values, 
        log_probs, 
        rewards, 
        where_done):
        """
        """
        self.validate_agent_id(agent_id)
        env_batch_size = self.episodes[agent_id].size

        #
        # When using lstm networks, we need to save the hidden states
        # encountered during the rollouts. These will later be used to
        # initialize the hidden states when updating the models.
        # Note that we pass in empty arrays when not using lstm networks.
        #
        if self.using_lstm:

            actor_hidden  = self.actor.hidden_state[0].clone()
            actor_cell    = self.actor.hidden_state[1].clone()

            critic_hidden = self.critic.hidden_state[0].clone()
            critic_cell   = self.critic.hidden_state[1].clone()

            if where_done.size > 0:
                actor_zero_hidden, actor_zero_cell = \
                    self.actor.get_zero_hidden_state(
                        batch_size = env_batch_size,
                        device     = self.device)

                actor_hidden[:, where_done, :] = \
                    actor_zero_hidden[:, where_done, :]

                actor_cell[:, where_done, :] = \
                    actor_zero_cell[:, where_done, :]

                critic_zero_hidden, critic_zero_cell = \
                    self.critic.get_zero_hidden_state(
                        batch_size = env_batch_size,
                        device     = self.device)

                critic_hidden[:, where_done, :] = \
                    critic_zero_hidden[:, where_done, :]

                critic_cell[:, where_done, :] = \
                    critic_zero_cell[:, where_done, :]

        else:
            empty_shape = (0, env_batch_size, 0)

            actor_hidden, actor_cell, critic_hidden, critic_cell  = \
                (np.empty(empty_shape),
                 np.empty(empty_shape),
                 np.empty(empty_shape),
                 np.empty(empty_shape))

        for ei_idx in range(env_batch_size):
            self.episodes[agent_id][ei_idx].add_info(
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
        agent_id,
        env_idxs,
        episode_lengths,
        terminal,
        ending_values,
        ending_rewards,
        status_dict):
        """
        """
        self.validate_agent_id(agent_id)

        for idx, env_i in enumerate(env_idxs):
            self.episodes[agent_id][env_i].end_episode(
                ending_ts      = episode_lengths[env_i],
                terminal       = terminal[idx],
                ending_value   = ending_values[idx].item(),
                ending_reward  = ending_rewards[idx].item())

            self.dataset.add_episode(self.episodes[agent_id][env_i])

            #
            # If we're using a dynamic bs clip, we clip to the min/max
            # rewards from the episode. Otherwise, rely on the user
            # provided range.
            #
            bs_min, bs_max = self.get_bs_clip_range(
                self.episodes[agent_id][env_i].rewards,
                status_dict)

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

    def finalize_dataset(self):
        """
        """
        self.dataset.build()

    def clear_dataset(self):
        """
        """
        self.dataset  = None
        self.episodes = {}

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

        if self.action_dtype in ["discrete", "multi-discrete"]:
            action = torch.tensor(action,
                dtype=torch.long).to(self.device)

        elif self.action_dtype in ["continuous", "multi-binary"]:
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

    def get_bs_clip_range(self, ep_rewards, status_dict):
        """
            Get the current bootstrap clip range.

            Arguments:
                ep_rewards    A numpy array containing the rewards for
                              this episode.
                status_dict   The status dictionary.

            Returns:
                A tuple containing the min and max values for the bootstrap
                clip.
        """
        if self.dynamic_bs_clip and ep_rewards is not None:
            bs_min = min(ep_rewards)
            bs_max = max(ep_rewards)

        else:
            iteration = status_dict["general"]["iteration"]
            timestep  = status_dict["general"]["timesteps"]

            bs_min = self.bootstrap_clip[0](
                iteration = iteration,
                timestep  = timestep)

            bs_max = self.bootstrap_clip[1](
                iteration = iteration,
                timestep  = timestep)

        return (bs_min, bs_max)

    def get_cloned_intrinsic_reward(
        self,
        obs,
        obs_augment):
        """
        """
        if obs_augment:
            _, clone_action, _ = self.get_action(obs[0:1])
            env_batch_size = obs[0:1].shape[0]
        else:
            _, clone_action, _ = self.get_action(obs)
            env_batch_size = obs.shape[0]

        clone_prev_obs = obs.copy()
        cloned_env = deepcopy(self.env)
        clone_obs, _, _, clone_info = cloned_env.step(clone_action)
        del cloned_env

        if obs_augment:
            action_shape = (env_batch_size,) + clone_action.shape[1:]
            clone_action = np.tile(clone_action.flatten(), env_batch_size)
            clone_action = clone_action.reshape(action_shape)

        intr_reward = self.get_intrinsic_reward(
            clone_prev_obs,
            clone_obs,
            clone_action)

        return intr_reward

    def save(self, save_path):
        """
        """
        policy_dir = "{}-policy".format(self.name)
        policy_save_path = os.path.join(save_path, policy_dir)

        if rank == 0 and not os.path.exists(policy_save_path):
            os.makedirs(policy_save_path)

        self.actor.save(policy_save_path)
        self.critic.save(policy_save_path)

        if self.enable_icm:
            self.icm_model.save(policy_save_path)

    def load(self, load_path):
        """
        """
        policy_dir = "{}-policy".format(self.name)
        policy_load_path = os.path.join(load_path, policy_dir)

        self.actor.load(policy_load_path)
        self.critic.load(policy_load_path)

        if self.enable_icm:
            self.icm_model.load(policy_load_path)

    def eval(self):
        """
        """
        self.actor.eval()
        self.critic.eval()

        if self.enable_icm:
            self.icm_model.eval()

    def train(self):
        """
        """
        self.actor.train()
        self.critic.train()

        if self.enable_icm:
            self.icm_model.train()

    def __getstate__(self):
        """
            Override the getstate method for pickling. We only want to keep
            things that won't upset pickle. The environment is something
            that we can't guarantee can be pickled.

            Returns:
                The state dictionary minus the environment.
        """
        state = self.__dict__.copy()
        del state["bootstrap_clip"]
        return state

    def __setstate__(self, state):
        """
            Override the setstate method for pickling.

            Arguments:
                The state loaded from a pickled PPO object.
        """
        self.__dict__.update(state)
        self.bootstrap_clip = (None, None)

    def __eq__(self, other):
        """
        """
        #
        # TODO: we currently don't compare optimizers because that
        # requires extra effort, and our current implementation will
        # enforce they're equal when the learning rate is equal.
        # We should update this at some point.
        #
        # FIXME: bootstrap clip is difficult to compare without using
        # functions that define __eq__, so we're skipping it.
        #
        is_equal = (
            isinstance(other, AgentPolicy)
            and self.action_space       == other.action_space
            and self.actor_obs_space    == other.actor_obs_space
            and self.critic_obs_space   == other.critic_obs_space
            and self.enable_icm         == other.enable_icm
            and self.intr_reward_weight == other.intr_reward_weight
            and self.test_mode          == other.test_mode
            and self.use_gae            == other.use_gae
            and self.gamma              == other.gamma
            and self.lambd              == other.lambd
            and self.dynamic_bs_clip    == other.dynamic_bs_clip
            and self.using_lstm         == other.using_lstm
            and self.act_dim            == other.act_dim
            and self.action_dtype       == other.action_dtype)

        return is_equal

    def __str__(self):
        """
        """
        str_self  = "AgentPolicy:\n"
        str_self += "    action space: {}\n".format(self.action_space)
        str_self += "    actor obs space: {}\n".format(self.actor_obs_space)
        str_self += "    critic obs space: {}\n".format(self.critic_obs_space)
        str_self += "    enable icm: {}\n".format(self.enable_icm)
        str_self += "    intr reward weight: {}\n".format(self.intr_reward_weight)
        str_self += "    test mode: {}\n".format(self.test_mode)
        str_self += "    use gae: {}\n".format(self.use_gae)
        str_self += "    gamma: {}\n".format(self.gamma)
        str_self += "    lambd: {}\n".format(self.lambd)
        str_self += "    dynamic bs clip: {}\n".format(self.dynamic_bs_clip)
        str_self += "    bootstrap clip: {}\n".format(self.bootstrap_clip)
        str_self += "    using lstm: {}\n".format(self.using_lstm)
        str_self += "    act dim: {}\n".format(self.act_dim)
        str_self += "    action dtype: {}\n".format(self.action_dtype)
        str_self += "    actor optim: {}\n".format(self.actor_optim)
        str_self += "    critic optim: {}\n".format(self.critic_optim)
        str_self += "    icm optim: {}\n".format(self.icm_optim)
        return str_self

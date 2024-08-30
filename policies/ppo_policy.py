import numpy as np
import os
import torch
from copy import deepcopy
from functools import reduce
from torch.optim import Adam
from torch import nn
from ppo_and_friends.utils.episode_info import EpisodeInfo, PPODataset, PPOSharedEpisodeDataset
from ppo_and_friends.networks.ppo_networks.icm import ICM
from ppo_and_friends.utils.mpi_utils import rank_print
from ppo_and_friends.utils.misc import get_space_dtype_str
import gymnasium.spaces as spaces
from ppo_and_friends.utils.mpi_utils import broadcast_model_parameters, mpi_avg_gradients
from ppo_and_friends.utils.misc import update_optimizer_lr
from ppo_and_friends.networks.ppo_networks.feed_forward import FeedForwardNetwork
from ppo_and_friends.networks.actor_critic.wrappers import to_actor, to_critic
from ppo_and_friends.utils.schedulers import LinearScheduler, CallableValue
from ppo_and_friends.utils.misc import get_flattened_space_length, get_action_prediction_shape
from ppo_and_friends.utils.spaces import FlatteningTuple, gym_space_to_gymnasium_space

from mpi4py import MPI
comm      = MPI.COMM_WORLD
rank      = comm.Get_rank()
num_procs = comm.Get_size()

class PPOPolicy():
    """
    A class representing a policy. A policy can be
    used by more than one agent, and more than one policy
    can exist in a given learning environment.
    """

    def __init__(self,
                 name,
                 action_space,
                 actor_observation_space,
                 critic_observation_space,
                 envs_per_proc,
                 bootstrap_clip      = (-100., 100.),
                 ac_network          = FeedForwardNetwork,
                 actor_kw_args       = {},
                 critic_kw_args      = {},
                 icm_kw_args         = {},
                 target_kl           = 100.,
                 surr_clip           = 0.2,
                 vf_clip             = None,
                 gradient_clip       = 0.5,
                 lr                  = 3e-4,
                 icm_lr              = 3e-4,
                 entropy_weight      = 0.01,
                 kl_loss_weight      = 0.0,
                 use_gae             = True,
                 gamma               = 0.99,
                 lambd               = 0.95,
                 dynamic_bs_clip     = False,
                 enable_icm          = False,
                 agent_shared_icm    = False,
                 icm_network         = ICM,
                 intr_reward_weight  = 1.0,
                 icm_beta            = 0.8,
                 use_huber_loss      = False,
                 test_mode           = False,
                 verbose             = False,
                 **kw_args):
        """
        Parameters:
        ----------
        name: str
            The name of this policy.
        action_space: gymnasium space
            The action space of this policy.
        actor_observation_space: gymnasium space
            The actor's observation space.
        critic_observation_space: gymnasium space
            The critic's observation space.
        envs_per_proc: int
            The number of environments each process owns. This means that
            a single step in the environment is actually stepping in
            envs_per_proc instances of the environment behind the scenes.
        bootstrap_clip: tuple or callable.
            When using GAE, we bootstrap the values
            and rewards when an epsiode is cut off
            before completion. In these cases, we
            clip the bootstrapped reward to a
            specific range. Why is this? Well, our
            estimated reward (from our value network)
            might be way outside of the expected
            range. We also allow the range min/max
            to be callables from utils/schedulers.
        ac_network: PPONetwork
            The type of network to use for the actor
            and critic.
        actor_kw_args: dict
            Keyword arguments for the actor network.
        critic_kw_args: dict
            Keyword arguments for the critic network.
        icm_kw_args: dict
            Keyword arguments for the icm network.
        target_kl: float
            KL divergence used for early stopping.
            This is typically set in the range
            [0.1, 0.5]. Use high values to disable.
        surr_clip: float
            The clip value applied to the surrogate
            (standard PPO approach).
        vf_clip: float
            An optional clip parameter used for
            clipping the value function loss.
        gradient_clip: float
            An optional clip value to use on the
            gradient update.
        lr: float or scheduler
            The learning rate. Can be
            a number or a scheduler class from
            utils/schedulers.py.
        icm_lr: float or scheduler
            The learning rate. Can be
            a number or a scheduler class from
            utils/schedulers.py.
        entropy_weight: float or scheduler
            The entropy weight. Can be
            a number or a scheduler class from
            utils/schedulers.py.
        kl_loss_weight: float
            A "kl coefficient" when adding kl
            divergence to the actor's loss. This
            is only used when > 0.0, and is off
            by default.
        use_gae: bool
            Should we use Generalized Advantage
            Estimations? If not, fall back on the
            vanilla advantage calculation.
        gamma: float
            The gamma parameter used in calculating
            rewards to go.
        lambd: float
            The 'lambda' value for calculating GAEs.
        dynamic_bs_clip: bool
            If set to True, bootstrap_clip will be
            used as the initial clip values, but all
            values thereafter will be taken from the
            global min and max rewards that have been
            seen so far.
        enable_icm: bool
            Enable ICM?
        agent_shared_icm: bool
            Enable agent sharing of observations and actions when using ICM?
        icm_network: PPONetwork
            The network to use for ICM applications.
        intr_reward_weight: float
            When using ICM, this weight will be
            applied to the intrinsic reward.
            Can be a number or a class from
            utils/schedulers.py.
        icm_beta: float
            The beta value used within the ICM.
        use_huber_loss: bool
            Should we use huber loss during the PPO update?
        test_mode: bool
            Are we in test mode?
        verbose: bool
            Enable verbosity?
        """
        self.name                   = name
        self.action_space           = action_space
        self.actor_obs_space        = actor_observation_space
        self.critic_obs_space       = critic_observation_space
        self.enable_icm             = enable_icm
        self.agent_shared_icm       = agent_shared_icm
        self.test_mode              = test_mode
        self.use_gae                = use_gae
        self.gamma                  = gamma
        self.lambd                  = lambd
        self.dynamic_bs_clip        = dynamic_bs_clip
        self.using_lstm             = False
        self.dataset                = None
        self.device                 = torch.device("cpu")
        self.agent_ids              = np.array([])
        self.episodes               = {}
        self.icm_beta               = icm_beta
        self.target_kl              = target_kl
        self.surr_clip              = surr_clip
        self.vf_clip                = vf_clip
        self.gradient_clip          = gradient_clip
        self.kl_loss_weight         = kl_loss_weight
        self.envs_per_proc          = envs_per_proc
        self.agent_grouping         = False
        self.have_step_constraints  = False
        self.have_reset_constraints = False
        self.verbose                = verbose
        self.use_huber_loss         = use_huber_loss
        self.frozen                 = False

        if callable(lr):
            self.lr = lr
        else:
            self.lr = CallableValue(lr)

        if callable(icm_lr):
            self.icm_lr = icm_lr
        else:
            self.icm_lr = CallableValue(icm_lr)

        if callable(entropy_weight):
            self.entropy_weight = entropy_weight
        else:
            self.entropy_weight = CallableValue(entropy_weight)

        if callable(intr_reward_weight):
            self.intr_reward_weight = intr_reward_weight
        else:
            self.intr_reward_weight = CallableValue(intr_reward_weight)

        #
        # Check for any old-gym spaces and convert them to gymnasium.
        #
        self.action_space     = gym_space_to_gymnasium_space(self.action_space)
        self.actor_obs_space  = gym_space_to_gymnasium_space(self.actor_obs_space)
        self.critic_obs_space = gym_space_to_gymnasium_space(self.critic_obs_space)

        self.action_dtype = get_space_dtype_str(self.action_space)

        #
        # If we've been given Tuple spaces, we need to convert them to
        # FlatteningTuple.
        #
        if (self.action_dtype == "mixed" and
            issubclass(type(self.action_space), spaces.Tuple)):
            self.action_space = FlatteningTuple(self.action_space.spaces)

        if (get_space_dtype_str(self.actor_obs_space) == "mixed" and
            issubclass(type(self.actor_obs_space), spaces.Tuple)):
            self.actor_obs_space = FlatteningTuple(self.actor_obs_space.spaces)

        if (get_space_dtype_str(self.critic_obs_space) == "mixed" and
            issubclass(type(self.critic_obs_space), spaces.Tuple)):
            self.critic_obs_space = FlatteningTuple(self.critic_obs_space.spaces)

        if self.action_dtype == "unknown":
            msg  = "ERROR: unknown action type: "
            msg += f"{type(self.action_space)} with dtype "
            msg += f"{self.action_space.dtype}."
            rank_print(msg)
            comm.Abort()
        else:
            rank_print("{} policy using {} actions.".format(
                self.name, self.action_dtype))

        #
        # One or both of our bootstrap clip ends might be a function of
        # our iteration.
        # We turn them both into functions for sanity.
        #
        min_bs_callable  = None
        max_bs_callable  = None
        bs_clip_callable = False

        self.have_bootstrap_clip = bootstrap_clip is not None

        if self.have_bootstrap_clip:
            if callable(bootstrap_clip[0]):
                min_bs_callable  = bootstrap_clip[0]
                bs_clip_callable = True
            else:
                min_bs_callable = CallableValue(bootstrap_clip[0])

            if callable(bootstrap_clip[1]):
                max_bs_callable  = bootstrap_clip[1]
                bs_clip_callable = True
            else:
                max_bs_callable = CallableValue(bootstrap_clip[1])

            self.bootstrap_clip = (min_bs_callable, max_bs_callable)

            if bs_clip_callable and dynamic_bs_clip:
                msg  = "WARNING: it looks like you've enabled dynamic_bs_clip "
                msg += "and also set the bootstrap clip to be callables. This is "
                msg += "redundant, and the dynamic clip will override the given "
                msg += "functions."
                rank_print(msg)
        else:
            self.bootstrap_clip = None

        self.action_dim       = get_flattened_space_length(self.action_space)
        self.action_pred_size = get_action_prediction_shape(self.action_space)[0]

        if self.verbose:
            msg  = f"Policy {self.name} is using action dim {self.action_dim} "
            msg += f"and action prediction size {self.action_pred_size}"
            rank_print(msg)

        self.network_args = {
        "ac_network"     : ac_network,
        "enable_icm"     : enable_icm,
        "icm_network"    : icm_network,
        "actor_kw_args"  : actor_kw_args,
        "critic_kw_args" : critic_kw_args,
        "icm_kw_args"    : icm_kw_args, 
        }
        self.network_args.update(kw_args)

    def finalize(self, status_dict, device):
        """
        Perfrom any finalizing tasks before we start using the policy.

        Parameters:
        -----------
        status_dict: dict
            The status dict for training.
        """
        #
        # Keep the original ordering around in case we need it after
        # shuffling.
        #
        self.agent_idxs = np.arange(len(self.agent_ids))
        self.num_agents = self.agent_idxs.size

        if self.agent_shared_icm and not self.agent_grouping:
            msg  = "ERROR: agent_shared_icm is only applicable to policies that "
            msg += "use agent grouping."
            rank_print(msg)
            comm.Abort()

        self._initialize_networks(**self.network_args)
        self.to(device)

        self.lr.finalize(status_dict)
        self.icm_lr.finalize(status_dict)
        self.entropy_weight.finalize(status_dict)
        self.intr_reward_weight.finalize(status_dict)

        if self.have_bootstrap_clip:
            self.bootstrap_clip[0].finalize(status_dict)
            self.bootstrap_clip[1].finalize(status_dict)

        self.actor_optim  = Adam(
            self.actor.parameters(), lr=self.lr(), eps=1e-5)
        self.critic_optim = Adam(
            self.critic.parameters(), lr=self.lr(), eps=1e-5)

        if self.enable_icm:
            self.icm_optim = Adam(self.icm_model.parameters(),
                lr=self.icm_lr(), eps=1e-5)
        else:
            self.icm_optim = None

    def seed(self, seed):
        """
        Set random seeds.
        """
        self.action_space.seed(seed)
        self.actor_obs_space.seed(seed)
        self.critic_obs_space.seed(seed)

    def register_agent(self, agent_id):
        """
        Register an agent with this policy.

        Parameters:
        -----------
        agent_id: str
            The id of the agent to register.
        """
        self.agent_ids = np.array(list(set(self.agent_ids).union({agent_id})))
        self.agent_ids = self.agent_ids

    def shuffle_agent_ids(self):
        """
        Shuffle our agent ids.
        """
        np.random.shuffle(self.agent_idxs)
        self.agent_ids = self.agent_ids[self.agent_idxs]

    def to(self, device):
        """
        Send this policy to a specified device.

        Parameters:
        -----------
        device: torch.device
            The device to send this policy to.
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
        icm_kw_args,
        **kw_args):
        """
        Initialize our networks.

        Parameters:
        -----------
        ac_network: class of type PPONetwork
            The network to use for the actor and critic.
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
        """
        #
        # Initialize our networks: actor, critic, and possibly ICM.
        #
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
        self.actor = to_actor(ac_network)(
            name         = "actor", 
            obs_space    = self.actor_obs_space,
            out_init     = 0.01,
            action_space = self.action_space,
            test_mode    = self.test_mode,
            **actor_kw_args)

        self.critic = to_critic(ac_network)(
            name         = "critic", 
            obs_space    = self.critic_obs_space,
            out_init     = 1.0,
            test_mode    = self.test_mode,
            **critic_kw_args)

        self.actor  = self.actor.to(self.device)
        self.critic = self.critic.to(self.device)

        if self.verbose:
            rank_print("")
            rank_print(f"Networks for {self.name} policy")
            rank_print(f"Actor network:\n{self.actor}")
            rank_print(f"Critic network:\n{self.critic}")

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

    def initialize_episodes(self, env_batch_size, status_dict):
        """
        Initialize episodes for rollout collection. This should be called
        at the start of a rollout.

        Parameters:
        -----------
        env_batch_size: int
            The number of environments per processor.
        status_dict: dict
            The status dictionary.
        """
        #
        # NOTE that different agents will map to different polices, meaning
        # that our dictionaries can be different sizes for each policy, but
        # the number of environment instances will be consistent.
        #
        self.episodes = {}
        for agent_id in self.agent_ids:
            self.episodes[agent_id] = np.array([None] * env_batch_size,
                dtype=object)

            bs_clip_range = self.get_bs_clip_range(None)

            for ei_idx in range(env_batch_size):
                self.episodes[agent_id][ei_idx] = EpisodeInfo(
                    starting_ts    = 0,
                    use_gae        = self.use_gae,
                    gamma          = self.gamma,
                    lambd          = self.lambd,
                    bootstrap_clip = bs_clip_range)

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

        self.dataset = PPODataset(
            device          = self.device,
            action_dtype    = self.action_dtype,
            sequence_length = sequence_length)

    def validate_agent_id(self, agent_id):
        """
        Assert that a given agent id is associated with this policy.
        This will Abort if the id is invalid.

        Parameters:
        -----------
        agent_id: str
            The agent id in question.
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
        critic_observations,
        observations,
        next_observations,
        raw_actions, 
        actions, 
        values, 
        log_probs, 
        rewards, 
        where_done):
        """
        Log information about a single step in an episode during a rollout.
        Note that our observaionts, etc, will be batched across environment
        instances, so we can have multiple observations for a single step.

        Parameters:
        -----------
        agent_id: str
            The agent id that this episode info is associated with.
        critic_observations: np.ndarray
            The critic observation(s).
        observations: np.ndarray
            The actor observation(s).
        next_observations: np.ndarray
            The actor observation(s.
        raw_actions: np.ndarray
            The raw action(s).
        actions: np.ndarray
            The actions(s) taken in the environment(s).
        values: np.darray
            The value(s) from our critic.
        log_probs: np.ndarray
            The log_prob(s) of our action distribution.
        rewards: np.ndarray
            The reward(s) received.
        where_done: np.ndarray
            Indicies mapping to which environments are done.
        """
        self.validate_agent_id(agent_id)

        #
        # NOTE: all agents should have the same env_batch_size.
        #
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
                critic_observation = critic_observations[ei_idx],
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
        ending_rewards):
        """
        End a rollout episode.

        Parameters:
        -----------
        agent_id: string
            The associated agent id.
        env_idxs: array-like
            The associated environment indices.
        episode_lengths: array-like
            The lenghts of the ending episode(s).
        terminal: array-like
            Which episodes are terminally ending.
        ending_values: array-like
            Ending values for the episode(s).
        ending_rewards: array-like
            Ending rewards for the episode(s)
        """
        if self.frozen:
            return

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
            bs_clip_range = self.get_bs_clip_range(
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
                bootstrap_clip = bs_clip_range)

    def finalize_dataset(self):
        """
        Build our rollout dataset. This should be called after
        the rollout has taken place and before training begins.
        """
        self.dataset.build()

    def clear_dataset(self):
        """
        Clear existing datasets. This should be called before
        a new rollout.
        """
        self.dataset  = None
        self.episodes = {}

    def get_rollout_actions(self, obs):
        """
        Given observations from our environment, determine what the
        next actions should be taken while allowing natural exploration.

        This method is explicitly meant to be used in training and will
        return more than just the environment actions.

        Parameters:
        -----------
        obs: array-like
            The environment observations.

        Returns:
        --------
        tuple:
            A tuple of form (raw_action, action, log_prob) s.t. "raw_action"
            is the distribution sample before any "squashing" takes place,
            "action" is the the action value that should be fed to the
            environment, and log_prob is the log probabilities from our
            probability distribution.
        """
        if len(obs.shape) < 2:
            msg  = "ERROR: get_rollout_actions expects a "
            msg += "batch of observations but "
            msg += "instead received shape {}.".format(obs.shape)
            rank_print(msg)
            comm.Abort()

        t_obs = torch.tensor(obs, dtype=torch.float32).to(self.device)

        if torch.isnan(t_obs).any():
            msg  = "ERROR: get_rollout_actions received observations "
            msg += "containing nan values!"
            msg += f"\n{t_obs}"
            rank_print(msg)
            comm.Abort()

        with torch.no_grad():
            action_pred = self.actor(t_obs)

        action_pred = action_pred.cpu().detach()

        if torch.isnan(action_pred).any():
            msg  = "ERROR: action prediction contains nan values!"
            msg += f"\nactions: {action_pred}"
            msg += f"\nobservations: {t_obs}"
            msg += f"\nobservations range: {t_obs.min()}, {t_obs.max()}"
            rank_print(msg)
            comm.Abort()

        dist = self.actor.distribution.get_distribution(action_pred)

        #
        # Our distribution gives us two potentially distinct actions, one of
        # which is guaranteed to be a raw sample from the distribution. The
        # other might be altered in some way (usually enforcing a range).
        #
        action, raw_action = self.actor.distribution.sample_distribution(dist)

        log_prob = self.actor.distribution.get_log_probs(dist, raw_action)

        raw_action = raw_action.detach().numpy()
        action     = action.detach().numpy()

        return raw_action, action, log_prob.detach()

    def get_inference_actions(self, obs, deterministic):
        """
        Given observations from our environment, determine what the
        actions should be.

        This method is meant to be used for inference only, and it
        will return the environment actions alone.

        Parameters:
        -----------
        obs: dict
            The environment observation.
        deterministic: bool
            If True, the action will always come from the highest
            probability action. Otherwise, our actions come from
            sampling the distribution.

        Returns:
        --------
        np.ndarray
            Predicted actions to perform in the environment.
        """
        if deterministic:
            return self._get_deterministic_actions(obs)
        return self._get_actions(obs)

    def _get_actions(self, obs):
        """
        Given observations from our environment, determine what the
        next actions should be taken while allowing natural exploration.

        Parameters:
        -----------
        obs: dict
            The environment observations.

        Returns:
        --------
        tuple:
            A tuple of form (raw_action, action, log_prob) s.t. "raw_action"
            is the distribution sample before any "squashing" takes place,
            "action" is the the action value that should be fed to the
            environment, and log_prob is the log probabilities from our
            probability distribution.
        """
        if len(obs.shape) < 2:
            msg  = "ERROR: _get_actions expects a "
            msg ++ "batch of observations but "
            msg += "instead received shape {}.".format(obs.shape)
            rank_print(msg)
            comm.Abort()

        t_obs = torch.tensor(obs, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            action_pred = self.actor(t_obs)

        action_pred = action_pred.cpu().detach()
        dist        = self.actor.distribution.get_distribution(action_pred)

        #
        # Our distribution gives us two potentially distinct actions, one of
        # which is guaranteed to be a raw sample from the distribution. The
        # other might be altered in some way (usually to enforce a range).
        #
        action, _ = self.actor.distribution.sample_distribution(dist)
        return action

    def _get_deterministic_actions(self, obs):
        """
        Given observations from our environment, determine what the
        next actions should be while not allowing any exploration.

        Parameters:
        -----------
        obs: dict
            The environment observations.

        Returns:
        --------
        np.ndarray
            The next actions to perform.
        """
        if len(obs.shape) < 2:
            msg  = "ERROR: _get_deterministic_actions expects a "
            msg ++ "batch of observations but "
            msg += "instead received shape {}.".format(obs.shape)
            rank_print(msg)
            comm.Abort()

        t_obs = torch.tensor(obs, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            return self.actor.get_refined_prediction(t_obs)

    def evaluate(self, batch_critic_obs, batch_obs, batch_actions):
        """
        Given a batch of observations, use our critic to approximate
        the expected return values. Also use a batch of corresponding
        actions to retrieve some other useful information.

        Parameters:
        -----------
        batch_critic_obs: torch.tensor
            A batch of observations for the critic.
        batch_obs: torch.tensor
            A batch of standard observations.
        batch_actions: torch.tensor
            A batch of actions corresponding to the batch of observations.

        Returns:
        --------
        tuple:
            A tuple of form (values, log_probs, entropies) s.t. values are
            the critic predicted value, log_probs are the log probabilities
            from our probability distribution, and entropies are the
            entropies from our distribution.
        """
        if torch.isnan(batch_critic_obs).any():
            msg  = "ERROR: evaluate received observations "
            msg += "containing nan values!"
            msg += f"\n{batch_critic_obs}"
            rank_print(msg)
            comm.Abort()

        values = self.critic(batch_critic_obs).squeeze()

        if torch.isnan(values).any():
            msg  = "ERROR: evaluate value prediction contains nan values!"
            msg += f"\nvalues: {values}"
            msg += f"\nobservations range: {batch_critic_obs.min()}, {batch_critic_obs.max()}"
            rank_print(msg)
            comm.Abort()

        action_pred = self.actor(batch_obs).cpu()

        if torch.isnan(action_pred).any():
            msg  = "ERROR: evaluate action prediction contains nan values!"
            msg += f"\nactions: {action_pred}"
            msg += f"\nobservations range: {batch_obs.min()}, {batch_obs.max()}"
            rank_print(msg)
            comm.Abort()

        dist = self.actor.distribution.get_distribution(action_pred)

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

        Parameters:
        -----------
        prev_obs: np.ndarray
            The previous observation (before the latest action).
        obs: np.ndarray
            The current observation.
        action: np.ndarray
            The action taken.
        """
        if len(obs.shape) < 2:
            msg  = "ERROR: get_intrinsic_reward expects a batch of "
            msg += "observations but "
            msg += "instead received shape {}.".format(obs.shape)
            rank_print(msg)
            comm.Abort()

        obs_1 = torch.tensor(prev_obs,
            dtype=torch.float32).to(self.device)
        obs_2 = torch.tensor(obs,
            dtype=torch.float32).to(self.device)

        if self.action_dtype in ["discrete", "multi-discrete"]:
            action = torch.tensor(action,
                dtype=torch.long).to(self.device)

        elif self.action_dtype in ["continuous", "multi-binary", "mixed"]:
            action = torch.tensor(action,
                dtype=torch.float32).to(self.device)

        else:
            msg  = f"ERROR: unknown action dtype of {self.action_dtype} "
            msg += "encountered when getting intrinsic reward."
            rank_print(msg)
            comm.Abort()

        if len(action.shape) != 2:
            action = action.unsqueeze(1)

        with torch.no_grad():
            intr_reward, _, _ = self.icm_model(obs_1, obs_2, action)

        batch_size   = obs.shape[0]
        intr_reward  = intr_reward.detach().cpu().numpy()
        intr_reward  = intr_reward.reshape((batch_size, -1))
        intr_reward *= self.intr_reward_weight()

        return intr_reward

    def get_agent_shared_intrinsic_rewards(*args):
        raise NotImplementedError

    def update_weights(self, actor_loss, critic_loss):
        """
        Update the weights of our actor/critic class.

        Parameters:
        -----------
        actor_loss: torch tensor
            The total loss for our actor.
        critic_loss: torch tensor
            The total loss for our critic.
        """
        if self.frozen:
            return

        #
        # Perform our backwards steps, and average gradients across ranks.
        #
        # arXiv:2005.12729v1 suggests that gradient clipping can
        # have a positive effect on training
        #
        self.actor_optim.zero_grad()
        actor_loss.backward(
            retain_graph = self.using_lstm)
        mpi_avg_gradients(self.actor)

        if self.gradient_clip is not None:
            nn.utils.clip_grad_norm_(
                self.actor.parameters(),
                self.gradient_clip)

        self.actor_optim.step()

        self.critic_optim.zero_grad()
        critic_loss.backward(
            retain_graph = self.using_lstm)

        mpi_avg_gradients(self.critic)

        if self.gradient_clip is not None:
            nn.utils.clip_grad_norm_(
                self.critic.parameters(),
                self.gradient_clip)

        self.critic_optim.step()

    def get_critic_values(self, obs):
        """
        Get the predicted values from the critic.

        Parameters:
        -----------
        obs: array-like
            An array of observations to predict values for.

        Returns:
        --------
        torch tensor:
            The predicted values.
        """
        return self.critic(obs)

    def update_learning_rate(self):
        """
        Update the learning rate.
        """
        if self.frozen:
            return

        update_optimizer_lr(self.actor_optim, self.lr())
        update_optimizer_lr(self.critic_optim, self.lr())

        if self.enable_icm:
            update_optimizer_lr(self.icm_optim, self.icm_lr())

    def get_bs_clip_range(self, ep_rewards):
        """
        Get the current bootstrap clip range.

        Parameters:
        -----------
        ep_rewards: numpy array or None.
            A numpy array containing the rewards for this episode.

        Returns:
        --------
        tuple or None:
            A tuple containing the min and max values for the bootstrap
            clip if we're using it. Otherwise, None.
        """
        if not self.have_bootstrap_clip:
            return None

        if self.dynamic_bs_clip and ep_rewards is not None:
            bs_min = min(ep_rewards)
            bs_max = max(ep_rewards)

        else:
            bs_min = self.bootstrap_clip[0]()
            bs_max = self.bootstrap_clip[1]()

        return (bs_min, bs_max)

    def apply_step_constraints(self, *args):
        """
        Apply any constraints needed when stepping through the environment.

        NOTE: This may alter the values returned by the environment.

        Parameters:
        -----------
        obs: dict
            Dictionary mapping agent ids to actor observations.
        critic_obs: dict
            Dictionary mapping agent ids to critic observations.
        reward: dict
            Dictionary mapping agent ids to rewards.
        terminated: dict
            Dictionary mapping agent ids to a termination flag.
        truncated: dict
            Dictionary mapping agent ids to a truncated flag.
        info: dict
            Dictionary mapping agent ids to info dictionaries.
        """
        return args

    def apply_reset_constraints(self, *args):
        """
        Apply any constraints needed when resetting the environment.

        NOTE: This may alter the values returned by the environment.

        Parameters:
        -----------
        obs: dict
            Dictionary mapping agent ids to actor observations.
        critic_obs: dict
            Dictionary mapping agent ids to critic observations.
        """
        return args

    def save(self, save_path, tag="latest"):
        """
        Save our policies and optimizers.

        Parameters:
        -----------
        save_path: str
            The state path to save the policy to.
        tag: str
            An optional tag directory to save the network to. This
            defaults to "latest".
        """
        if type(tag) != str:
            tag = str(tag)

        policy_dir = "{}-policy".format(self.name)
        policy_save_path = os.path.join(save_path, policy_dir, tag)

        if rank == 0 and not os.path.exists(policy_save_path):
            os.makedirs(policy_save_path)

        self._save_policies(policy_save_path)
        self._save_optimizers(policy_save_path)

    def load(self, load_path, tag="latest"):
        """
        Load our policies and optimizers.

        Parameters:
        -----------
        load_path: str
            The state path to load the policy from.
        tag: str
            An optional tag directory to load the network from. This
            defaults to "latest".
        """
        if type(tag) != str:
            tag = str(tag)

        policy_dir = "{}-policy".format(self.name)
        policy_load_path = os.path.join(load_path, policy_dir, tag)

        self._load_policies(policy_load_path)
        self._load_optimizers(policy_load_path)

    def _save_policies(self, save_path):
        """
        Save our policies.

        Parameters:
        -----------
        save_path: str
            The state path to save the policy to.
        """
        self.actor.save(save_path)
        self.critic.save(save_path)

        if self.enable_icm:
            self.icm_model.save(save_path)

    def _load_policies(self, load_path):
        """
        Load our policies.

        Parameters:
        -----------
        load_path: str
            The state path to load the policy from.
        """
        self.actor.load(load_path)
        self.critic.load(load_path)

        if self.enable_icm:
            self.icm_model.load(load_path)

    def _save_optimizers(self, save_path):
        """
        Save our optimizers.

        Parameters:
        -----------
        save_path: str
            The state path to save the optimizers to.
        """
        if self.test_mode:
            return

        actor_optim_f  = os.path.join(save_path, f"actor_optim_{rank}")
        critic_optim_f = os.path.join(save_path, f"critic_optim_{rank}")

        torch.save(self.actor_optim.state_dict(), actor_optim_f)
        torch.save(self.critic_optim.state_dict(), critic_optim_f)

        if self.enable_icm:
            icm_optim_f = os.path.join(save_path, f"icm_optim_{rank}")
            torch.save(self.icm_optim.state_dict(), icm_optim_f)

    def _load_optimizers(self, load_path):
        """
        Load our optimizers.

        Parameters:
        -----------
        load_path: str
            The state path to load the optimizers from.
        """
        if self.test_mode:
            return

        try:
            if self.test_mode:
                load_rank = 0
            else:
                load_rank = rank

            actor_optim_f = os.path.join(load_path, f"actor_optim_{load_rank}")
            if not os.path.exists(actor_optim_f):
                actor_optim_f = os.path.join(load_path, f"actor_optim_0")

            critic_optim_f = os.path.join(load_path, f"critic_optim_{load_rank}")
            if not os.path.exists(critic_optim_f):
                critic_optim_f = os.path.join(load_path, f"critic_optim_0")

            self.actor_optim.load_state_dict(torch.load(actor_optim_f))
            self.critic_optim.load_state_dict(torch.load(critic_optim_f))

            if self.enable_icm:
                icm_optim_f = os.path.join(load_path, f"icm_optim_{load_rank}")
                if not os.path.exists(icm_optim_f):
                    icm_optim_f = os.path.join(load_path, f"icm_optim_0")

                self.icm_optim.load_state_dict(torch.load(icm_optim_f))
        except Exception:
            rank_print("WARNING: unable to find saved optimizers to load. Skipping...")

    def direct_load(self, policy_load_path):
        """
        Load our policy directly from the provided path.

        Parameters:
        -----------
        policy_load_path: str
            The direct path to load the policy from.
        """
        self.actor.load(policy_load_path)
        self.critic.load(policy_load_path)

        if self.enable_icm:
            self.icm_model.load(policy_load_path)

    def eval(self):
        """
        Set the policy to evaluation mode.
        """
        self.actor.eval()
        self.critic.eval()

        if self.enable_icm:
            self.icm_model.eval()

    def train(self):
        """
        Set the policy to train mode.
        """
        self.actor.train()
        self.critic.train()

        if self.enable_icm:
            self.icm_model.train()

    def freeze(self):
        """
        Freeze the poicy so that its weights are not updated.
        """
        self.frozen = True

    def unfreeze(self):
        """
        Un-freeze the poicy so that its weights will be updated.
        """
        self.frozen = False

    def __getstate__(self):
        """
        Override the getstate method for pickling. We only want to keep
        things that won't upset pickle. The environment is something
        that we can't guarantee can be pickled.

        Returns:
        --------
        dict:
            The state dictionary minus the environment.
        """
        state = self.__dict__.copy()
        del state["bootstrap_clip"]
        return state

    def __setstate__(self, state):
        """
        Override the setstate method for pickling.

        Parameters:
        -----------
        state: dict
            The state loaded from a pickled PPO object.
        """
        self.__dict__.update(state)
        self.bootstrap_clip = None

    def __eq__(self, other):
        """
        Compare two policies.

        Parameters:
        -----------
        other: PPOPolicy object
            An instance of PPOPolicy.
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
            isinstance(other, PPOPolicy)
            and self.action_space       == other.action_space
            and self.actor_obs_space    == other.actor_obs_space
            and self.critic_obs_space   == other.critic_obs_space
            and self.enable_icm         == other.enable_icm
            and self.test_mode          == other.test_mode
            and self.use_gae            == other.use_gae
            and self.gamma              == other.gamma
            and self.lambd              == other.lambd
            and self.dynamic_bs_clip    == other.dynamic_bs_clip
            and self.using_lstm         == other.using_lstm
            and self.action_dtype       == other.action_dtype)

        return is_equal

    def __str__(self):
        """
        A string representation of the policy.
        """
        str_self  = "PPOPolicy:\n"
        str_self += "    action space: {}\n".format(self.action_space)
        str_self += "    actor obs space: {}\n".format(self.actor_obs_space)
        str_self += "    critic obs space: {}\n".format(self.critic_obs_space)
        str_self += "    enable icm: {}\n".format(self.enable_icm)
        str_self += "    intr reward weight: {}\n".format(self.intr_reward_weight)
        str_self += "    lr: {}\n".format(self.lr)
        str_self += "    icm_lr: {}\n".format(self.icm_lr)
        str_self += "    entropy_weight: {}\n".format(self.entropy_weight)
        str_self += "    test mode: {}\n".format(self.test_mode)
        str_self += "    use gae: {}\n".format(self.use_gae)
        str_self += "    gamma: {}\n".format(self.gamma)
        str_self += "    lambd: {}\n".format(self.lambd)
        str_self += "    dynamic bs clip: {}\n".format(self.dynamic_bs_clip)
        str_self += "    bootstrap clip: {}\n".format(self.bootstrap_clip)
        str_self += "    using lstm: {}\n".format(self.using_lstm)
        str_self += "    action dtype: {}\n".format(self.action_dtype)
        str_self += "    actor optim: {}\n".format(self.actor_optim)
        str_self += "    critic optim: {}\n".format(self.critic_optim)
        str_self += "    icm optim: {}\n".format(self.icm_optim)
        return str_self

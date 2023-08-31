import numpy as np
import os
import torch
from torch.nn import functional as t_func
from torch import nn
from copy import deepcopy
from functools import reduce
from torch.optim import Adam
from ppo_and_friends.utils.episode_info import EpisodeInfo, PPODataset, PPOSharedEpisodeDataset
from ppo_and_friends.networks.ppo_networks.icm import ICM
from ppo_and_friends.utils.mpi_utils import rank_print
from ppo_and_friends.utils.misc import get_action_dtype
from gymnasium.spaces import Box, Discrete, MultiDiscrete, MultiBinary
from ppo_and_friends.utils.mpi_utils import broadcast_model_parameters, mpi_avg_gradients
from ppo_and_friends.utils.misc import update_optimizer_lr
from ppo_and_friends.networks.ppo_networks.feed_forward import FeedForwardNetwork
import ppo_and_friends.networks.actor_critic.multi_agent_transformer as mat
from ppo_and_friends.utils.schedulers import LinearScheduler, CallableValue
from ppo_and_friends.utils.misc import get_size_and_shape
from ppo_and_friends.policies.agent_policy import AgentPolicy

from mpi4py import MPI
comm      = MPI.COMM_WORLD
rank      = comm.Get_rank()
num_procs = comm.Get_size()


class MATPolicy(AgentPolicy):
    """
    The Multi-Agent transformer policy. This implementation is based off of
    arXiv:2205.14953
    """

    def __init__(self,
                 ac_network       = mat.MATActorCritic,
                 mat_kw_args      = {},
                 shared_reward_fn = np.sum,
                 **kw_args):
        """
        Initialize the policy.

        Parameters:
        -----------
        ac_network: class
            The class to use for our actor/critic. This should be of type/subtype
            PPONetwork.
        shared_reward_fn: np function 
            This is a numpy "reduction" function that will be used to
            create shared rewards in environments that don't already have them.
            Current options are np.min, np.max, np.mean, np.max. MAT requires
            that all agents of a policy receive the same reward.
        """
        super(MATPolicy, self).__init__(
            ac_network       = ac_network,
            shared_reward_fn = shared_reward_fn,
            mat_kw_args      = mat_kw_args,
            **kw_args)

        self.agent_grouping = True

        #
        # MAT really only uses the critic observation space. So, if our
        # env is returning different spaces for actor and critic, we need
        # to put in some extra work to transfer the critic observations
        # to the actor observations.
        #
        if self.actor_obs_space != self.critic_obs_space:
            self.have_step_constraints  = True
            self.have_reset_constraints = True
            self.actor_obs_space        = self.critic_obs_space

        #FIXME: throw error if using_lstm == True.

        #
        # Shuffle agents right off the bat so that different processors
        # have different ordering.
        #
        self.shuffle_agent_ids()

    def _initialize_networks(
        self,
        ac_network,
        enable_icm,
        icm_network,
        mat_kw_args,
        icm_kw_args,
        **kw_args):
        """
        Initialize our networks.

        Parameters:
        -----------
        ac_network: class of type PPONetwork
            The network to use for the actor/critic.
        enable_icm: bool
            Whether or not to enable ICM.
        icm_network: class of type PPONetwork
            The network class to use for ICM (when enabled).
        mat_kw_args: dict
            Keyword args for the MAT network.
        icm_kw_args: dict
            Keyword args for the ICM network.
        """
        if not issubclass(ac_network, mat.MATActorCritic):
            msg  = "ERROR: ac_network for MATPolicy must be a "
            msg += "subtype of MATActorCritic, but received "
            msg += f"{ac_network}."
            rank_print(msg)
            comm.Abort()

        #
        # Initialize our networks: actor-critic, and possibly ICM.
        # NOTE: MAT uses the same observation space for both actor
        # and critic, and we default to the critic space to allow for
        # different observation views (local, policy, global).
        #
        self.actor_critic = ac_network(
            name         = "actor_critic", 
            obs_space    = self.critic_obs_space,
            action_space = self.action_space,
            num_agents   = len(self.agent_ids),
            test_mode    = self.test_mode,
            **mat_kw_args)

        self.actor_critic  = self.actor_critic.to(self.device)

        broadcast_model_parameters(self.actor_critic)
        comm.barrier()

        self.actor  = self.actor_critic.actor
        self.critic = self.actor_critic.critic

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
        self.dataset = PPOSharedEpisodeDataset(
            device          = self.device,
            action_dtype    = self.action_dtype,
            sequence_length = 1,
            num_envs        = self.envs_per_proc,
            agent_ids       = self.agent_ids)

    def finalize(self, status_dict):
        """
        Perfrom any finalizing tasks before we start using the policy.

        Parameters:
        -----------
        status_dict: dict
            The status dict for training.
        """
        self.lr.finalize(status_dict)
        self.icm_lr.finalize(status_dict)
        self.entropy_weight.finalize(status_dict)
        self.intr_reward_weight.finalize(status_dict)
        self.bootstrap_clip[0].finalize(status_dict)
        self.bootstrap_clip[1].finalize(status_dict)

        self.actor_critic_optim = Adam(
            self.actor_critic.parameters(), lr=self.lr(), eps=1e-5)

        if self.enable_icm:
            self.icm_optim = Adam(self.icm_model.parameters(),
                lr=self.icm_lr(), eps=1e-5)
        else:
            self.icm_optim = None

    def to(self, device):
        """
        Send this policy to a specified device.

        Parameters:
        -----------
        device: torch.device
            The device to send this policy to.
        """
        self.device       = device
        self.actor_critic = self.actor_critic.to(self.device)

        if self.enable_icm:
            self.icm_model = self.icm_model.to(self.device)

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
        Get a numpy "action block" that can be used for auto-regressive actions.
        The action block will be a numpy array of shape
        (batch_size, num_agents, action_pred_size). The first agent position
        will be a start token, and it is used to predict/evaluate the next
        (first) actions. All other agents will have empty (zeros) actions.

        Parameters:
        -----------
        batch_size: int
            The batch size for the action block.

        Returns:
        --------
        np.ndarray:
            An array of shape (batch_size, num_agents, action_pred_size) s.t.
            the first agent's action is a start token.
        """
        num_agents = len(self.agent_ids)

        if self.action_dtype == "continuous":
            action_block = torch.zeros(
               (batch_size, num_agents, self.action_pred_size)).to(self.device)

        elif self.action_dtype in ["discrete", "multi-discrete"]:
            action_block = torch.zeros(
                (batch_size, num_agents, self.action_pred_size + 1)).to(self.device)
            action_block[:, 0, 0] = 1

        return action_block

    def _evaluate_actions(self, obs, batch_actions):
        """
        """
        batch_size = obs.shape[0]
        num_agents = len(self.agent_ids)

        action_block = self._get_tokened_action_block(batch_size)

        if self.action_dtype == "discrete":
            #FIXME: Why are we adding the start token and removing the first
            # agent's actions?
            # another way around this might be to shuffle the agents during
            # training?
            action_block[:, 1:, 1:] = t_func.one_hot(batch_actions,
                num_classes=self.action_pred_size)[:, :-1, :]

        elif self.action_dtype == "multi-discrete":
            #FIXME: move to func
            one_hot_actions = torch.zeros((batch_size, num_agents, self.action_pred_size))
            start = 0
            for a_idx, dim in enumerate(self.action_space.nvec):
                stop = start + dim
                one_hot_actions[:, :, start : stop] = \
                    t_func.one_hot(batch_actions[:, :, a_idx], dim)

            action_block[:, 1:, 1:] = one_hot_actions[:, :-1, :]

        else:
            action_block[:, 1:, :] = batch_actions[:, :-1, :]

        values, action_pred = self.actor_critic(obs, action_block)

        if self.action_dtype == "multi-discrete":
            action_pred   = action_pred.reshape((-1, self.action_pred_size))
            batch_actions = batch_actions.reshape((-1, self.action_dim))

        dist    = self.actor.distribution.get_distribution(action_pred)
        entropy = self.actor.distribution.get_entropy(dist, action_pred)

        if self.actor.action_dtype == "continuous" and len(batch_actions.shape) < 2:
            log_probs = self.actor.distribution.get_log_probs(
                dist,
                batch_actions.unsqueeze(1).cpu())
        else:
            log_probs = self.actor.distribution.get_log_probs(
                dist,
                batch_actions.cpu())

        if self.action_dtype == "multi-discrete":
            action_pred = action_pred.reshape((batch_size,
                num_agents, self.action_pred_size))

            log_probs = log_probs.reshape((batch_size, num_agents, -1))
            entropy   = entropy.reshape((batch_size, num_agents, -1))

        return values, log_probs, entropy

    def _get_autoregressive_actions_with_exploration(self, encoded_obs):
        """
        """
        batch_size   = encoded_obs.shape[0]
        num_agents   = len(self.agent_ids)
        action_block = self._get_tokened_action_block(batch_size)

        if self.action_dtype in ["discrete", "multi-discrete"]:
            action_offset     = 1

            output_action     = torch.zeros((batch_size,
                num_agents, self.action_dim)).long()

            output_raw_action = torch.zeros_like(output_action).long()
        else:
            action_offset     = 0
            output_action     = torch.zeros((batch_size, num_agents, self.action_dim))
            output_action     = output_action.float()
            output_raw_action = torch.zeros_like(output_action).float()

        output_log_prob = torch.zeros_like(output_action, dtype=torch.float32)

        with torch.no_grad():
            for i in range(num_agents):
                action_pred = self.actor(action_block, encoded_obs)[:, i, :]
                dist        = self.actor.distribution.get_distribution(action_pred)

                action, raw_action = self.actor.distribution.sample_distribution(dist)
                log_prob = self.actor.distribution.get_log_probs(dist, raw_action)

                if self.action_dtype == "discrete":
                    output_action[:, i, :]     = action.unsqueeze(0)
                    output_raw_action[:, i, :] = raw_action.unsqueeze(0)
                    output_log_prob[:, i, :]   = log_prob.unsqueeze(0)
                    action = t_func.one_hot(action, num_classes=self.action_pred_size)

                elif self.action_dtype == "multi-discrete":
                    output_action[:, i, :]     = action
                    output_raw_action[:, i, :] = raw_action
                    output_log_prob[:, i, :]   = log_prob

                    #FIXME: move to function
                    action = action.flatten()
                    one_hot_action = torch.zeros(self.action_pred_size)
                    start = 0
                    for a_idx, dim in enumerate(self.action_space.nvec):
                        stop = start + dim
                        one_hot_action[start : stop] = t_func.one_hot(action[a_idx], dim)

                    action = one_hot_action 

                else:
                    output_action[:, i, :]     = action
                    output_raw_action[:, i, :] = raw_action
                    output_log_prob[:, i, :]   = log_prob

                if i + 1 < num_agents:
                    action_block[:, i + 1, action_offset:] = action

        return output_action, output_raw_action, output_log_prob

    def _get_autoregressive_actions_without_exploration(self, encoded_obs):
        """
        """
        batch_size   = encoded_obs.shape[0]
        num_agents   = len(self.agent_ids)
        action_block = self._get_tokened_action_block(batch_size)

        if self.action_dtype in ["discrete", "multi-discrete"]:
            action_offset = 1
            output_action = torch.zeros((batch_size, num_agents, 1)).long()
        else:
            action_offset = 0
            output_action = torch.zeros((batch_size, num_agents, self.action_dim))
            output_action = output_action.float()

        with torch.no_grad():
            for i in range(num_agents):
                action = self.actor.get_refined_prediction(
                    action_block, encoded_obs)

                if len(action.shape) < 3:
                    action = action.unsqueeze(-1)

                action = action[:, i, :]

                if self.action_dtype == "discrete":
                    output_action[:, i, :] = action.unsqueeze(0)
                    action = t_func.one_hot(action, num_classes=self.action_pred_size)

                elif self.action_dtype == "multi-discrete":
                    output_action[:, i, :] = action.unsqueeze(0)

                    # FIXME: create function for this.
                    action = action.flatten()
                    one_hot_action = torch.zeros(self.action_pred_size)
                    start = 0
                    for a_idx, dim in enumerate(self.action_space.nvec):
                        stop = start + dim
                        one_hot_action[start : stop] = t_func.one_hot(action[a_idx], dim)

                    action = one_hot_action 

                else:
                    output_action[:, i, :] = action

                if i + 1 < num_agents:
                    action_block[:, i + 1, action_offset:] = action

        return output_action

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

        #
        # The incoming shape is (num_agents, num_envs, obs_size). MAT wants
        # (num_envs, num_agents, obs_size).
        #
        obs = np.swapaxes(obs, 0, 1)
        t_obs = torch.tensor(obs, dtype=torch.float)
        t_obs = t_obs.to(self.device)

        encoded_obs, _ = self.critic(t_obs)
        actions, raw_actions, log_probs = \
            self._get_autoregressive_actions_with_exploration(encoded_obs)

        actions     = torch.swapaxes(actions, 0, 1)
        raw_actions = torch.swapaxes(raw_actions, 0, 1)
        log_probs   = torch.swapaxes(log_probs, 0, 1)

        actions     = actions.detach().numpy()
        raw_actions = raw_actions.detach().numpy()

        return raw_actions, actions, log_probs.detach()

    def evaluate(self, batch_critic_obs, batch_obs, batch_actions):
        """
        Given a batch of observations, use our critic to approximate
        the expected return values. Also use a batch of corresponding
        actions to retrieve some other useful information.

        Parameters:
        -----------
        batch_critic_obs: torch tensor
            A batch of observations for the critic.
        batch_obs: torch tensor
            A batch of standard observations.
        batch_actions: torch tensor
            A batch of actions corresponding to the batch of observations.

        Returns:
        --------
        tuple:
            A tuple of form (values, log_probs, entropies) s.t. values are
            the critic predicted value, log_probs are the log probabilities
            from our probability distribution, and entropies are the
            entropies from our distribution.
        """
        #
        # NOTE: when evaluating, our incoming data is coming from our
        # torch dataset, and it has already been re-shaped into the expected
        # (batch_size, num_agents, *) format.
        #
        values, log_probs, entropy = self._evaluate_actions(batch_critic_obs, batch_actions)

        return values, log_probs.to(self.device), entropy.to(self.device)

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
        _, values = self.critic(obs)
        return values

    def update_weights(self, actor_loss, critic_loss):
        """
        """
        actor_critic_loss = actor_loss + critic_loss

        self.actor_critic_optim.zero_grad()
        actor_critic_loss.backward()
        mpi_avg_gradients(self.actor_critic)

        if self.gradient_clip is not None:
            nn.utils.clip_grad_norm_(
                self.actor_critic.parameters(),
                self.gradient_clip)

        self.actor_critic_optim.step()

    def get_inference_actions(self, obs, explore):
        """
        Given observations from our environment, determine what the
        actions should be.

        This method is meant to be used for inference only, and it
        will return the environment actions alone.

        Parameters:
        -----------
        obs: dict
            The environment observation.
        explore: bool
            Should we allow exploration?

        Returns:
        --------
        dict:
            Predicted actions to perform in the environment.
        """
        if explore:
            return self._get_actions_with_exploration(obs)
        return self._get_actions_without_exploration(obs)

    def _get_actions_with_exploration(self, obs):
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
        if len(obs.shape) < 3:
            msg  = "ERROR: _get_action_with_exploration expects a "
            msg += "batch of agent observations but "
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

        encoded_obs, _ = self.critic(t_obs)
        actions, _, _ = \
            self._get_autoregressive_actions_with_exploration(encoded_obs)

        actions = torch.swapaxes(actions, 0, 1)

        return actions

    def _get_actions_without_exploration(self, obs):
        """
        Given observations from our environment, determine what the
        next actions should be while not allowing any exploration.

        Parameters:
        -----------
        obs: dict
            The environment observations.

        Returns:
        --------
        dict:
            The next actions to perform.
        """
        if len(obs.shape) < 3:
            msg  = "ERROR: _get_action_without_exploration expects a "
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

        encoded_obs, _ = self.critic(t_obs)

        actions = \
            self._get_autoregressive_actions_without_exploration(encoded_obs)

        actions = torch.swapaxes(actions, 0, 1)

        return actions

    def apply_step_constraints(
        self,
        obs,
        critic_obs,
        reward,
        terminated,
        truncated,
        info):
        """
        """
        if self.have_step_constraints:
            #
            # For MAT, the critic and actor are entwined. The observations that
            # are fed to the actor are first encoded by the critic, so they
            # need to see the same observations.
            #
            for agent_id in self.agent_ids:
                obs[agent_id] = critic_obs[agent_id]

                #
                # NOTE: when training, we always receive an ndarray. When testing,
                # it's a flat value.
                #
                if self.test_mode:
                    for env_idx in range(self.envs_per_proc):
                        if terminated[agent_id]:
                            info[agent_id]["terminal observation"] =\
                                info[agent_id]["terminal critic observation"]
                else:
                    for env_idx in range(self.envs_per_proc):
                        if terminated[agent_id][env_idx]:
                            info[agent_id][env_idx]["terminal observation"] =\
                                info[agent_id][env_idx]["terminal critic observation"]

        return obs, critic_obs, reward, terminated, truncated, info

    def apply_reset_constraints(
        self,
        obs,
        critic_obs):
        """
        """
        if self.have_reset_constraints:
            #
            # For MAT, the critic and actor are entwined. The observations that
            # are fed to the actor are first encoded by the critic, so they
            # need to see the same observations.
            #
            for agent_id in self.agent_ids:
                obs[agent_id] = critic_obs[agent_id]

        return obs, critic_obs

    def update_learning_rate(self):
        """
        Update the learning rate.
        """
        update_optimizer_lr(self.actor_critic_optim, self.lr())

        if self.enable_icm:
            update_optimizer_lr(self.icm_optim, self.icm_lr())

    def save(self, save_path):
        """
        Save our policy.

        Parameters:
        -----------
        save_path: str
            The path to save the policy to.
        """
        policy_dir = "{}-policy".format(self.name)
        policy_save_path = os.path.join(save_path, policy_dir)

        if rank == 0 and not os.path.exists(policy_save_path):
            os.makedirs(policy_save_path)

        comm.barrier()

        self.actor_critic.save(policy_save_path)

        if self.enable_icm:
            self.icm_model.save(policy_save_path)

    def load(self, load_path):
        """
        Load our policy.

        Parameters:
        -----------
        load_path: str
            The path to load the policy from.
        """
        policy_dir = "{}-policy".format(self.name)
        policy_load_path = os.path.join(load_path, policy_dir)

        self.actor_critic.load(policy_load_path)

        if self.enable_icm:
            self.icm_model.load(policy_load_path)

    def eval(self):
        """
        Set the policy to evaluation mode.
        """
        self.actor_critic.eval()

        if self.enable_icm:
            self.icm_model.eval()

    def train(self):
        """
        Set the policy to train mode.
        """
        self.actor_critic.train()

        if self.enable_icm:
            self.icm_model.train()


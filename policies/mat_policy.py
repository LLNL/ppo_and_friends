import numpy as np
import os
import torch
from torch.nn import functional as t_func
from torch import nn
from copy import deepcopy
from torch.optim import Adam
from ppo_and_friends.utils.episode_info import EpisodeInfo, PPODataset, PPOSharedEpisodeDataset
from ppo_and_friends.networks.ppo_networks.icm import ICM
from ppo_and_friends.utils.mpi_utils import rank_print
from ppo_and_friends.utils.mpi_utils import broadcast_model_parameters, mpi_avg_gradients
from ppo_and_friends.utils.misc import update_optimizer_lr
from ppo_and_friends.networks.ppo_networks.feed_forward import FeedForwardNetwork
import ppo_and_friends.networks.actor_critic.multi_agent_transformer as mat
from ppo_and_friends.policies.ppo_policy import PPOPolicy
from ppo_and_friends.utils.misc import get_agent_shared_space

from mpi4py import MPI
comm      = MPI.COMM_WORLD
rank      = comm.Get_rank()
num_procs = comm.Get_size()


class MATPolicy(PPOPolicy):
    """
    The Multi-Agent transformer policy. This implementation is based off of
    arXiv:2205.14953
    """

    def __init__(self,
                 ac_network       = mat.MATActorCritic,
                 mat_kw_args      = {},
                 use_huber_loss   = True,
                 **kw_args):
        """
        Initialize the policy.

        Parameters:
        -----------
        ac_network: class
            The class to use for our actor/critic. This should be of type/subtype
            PPONetwork.
        use_huber_loss: bool
            Should we use huber loss during the PPO update?
        """
        super(MATPolicy, self).__init__(
            ac_network       = ac_network,
            mat_kw_args      = mat_kw_args,
            use_huber_loss   = use_huber_loss,
            **kw_args)

        self.agent_grouping = True

        #
        # MAT really only uses the critic observation space. So, if our
        # env is returning different spaces for actor and critic, we need
        # to put in some extra work to transfer the critic observations
        # to the actor observations.
        #
        self.expanded_actor_space = False
        if self.actor_obs_space != self.critic_obs_space:
            self.expanded_actor_space   = True
            self.have_step_constraints  = True
            self.have_reset_constraints = True
            self.actor_obs_space        = self.critic_obs_space

        if self.using_lstm == True:
            msg  = "ERROR: MAT is not compatible with lstm, but using_lstm "
            msg += "is set to True. Bailing..."
            rank_print(msg)
            comm.Abort()

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

        if self.verbose:
            rank_print("")
            rank_print(f"Networks for {self.name} policy")
            rank_print(f"Actor network:\n{self.actor}")
            rank_print(f"Critic network:\n{self.critic}")

        if enable_icm:
            self.icm_agent_ids = None
            icm_obs_space      = self.actor_obs_space
            icm_action_space   = self.action_space

            #
            # MAT always uses the critic obsevations for both critic and actor.
            # We don't currently support agent_shared_icm when using global or
            # policy critic views because the original local actor observations
            # are not saved. It's tempting to just use the critic view, since it
            # already contains concatenated/shared observations in the local
            # and global case, but the agent ids are shuffled, which is not
            # what we want when using the agent_shared_icm feature.
            #
            if self.agent_shared_icm:
                if self.expanded_actor_space:
                    msg  = "ERROR: agent_shared_icm can only be enabled with "
                    msg += "the multi-agent transformer if critic view is set "
                    msg += f"to local."
                    rank_print(msg)
                    comm.Abort()

                #
                # Our actor and critic are both using local
                # obsevations. We need to expand them in this case.
                #
                icm_obs_space    = get_agent_shared_space(self.actor_obs_space, self.num_agents)
                icm_action_space = get_agent_shared_space(self.action_space, self.num_agents)

                #
                # We need to pass agents to ICM in a consistent order.
                # Since policies are allowed to shuffle their ids, we
                # make a copy of the original order and use this for ICM.
                #
                self.icm_agent_ids = self.agent_ids.copy()

            self.icm_model = icm_network(
                name         = "icm",
                obs_space    = icm_obs_space,
                action_space = icm_action_space,
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

    def finalize(self, status_dict, device):
        """
        Perfrom any finalizing tasks before we start using the policy.

        Parameters:
        -----------
        status_dict: dict
            The status dict for training.
        device: torch device
            The device to send our networks to.
        """
        #
        # Keep the original ordering around in case we need it after
        # shuffling.
        #
        self.agent_idxs = np.arange(len(self.agent_ids))
        self.num_agents = self.agent_idxs.size

        self._initialize_networks(**self.network_args)
        self.to(device)

        self.lr.finalize(status_dict)
        self.icm_lr.finalize(status_dict)
        self.entropy_weight.finalize(status_dict)
        self.intr_reward_weight.finalize(status_dict)

        if self.have_bootstrap_clip:
            self.bootstrap_clip[0].finalize(status_dict)
            self.bootstrap_clip[1].finalize(status_dict)

        self.actor_critic_optim = Adam(
            self.actor_critic.parameters(), lr=self.lr(), eps=1e-5)

        if self.enable_icm:
            self.icm_optim = Adam(self.icm_model.parameters(),
                lr=self.icm_lr(), eps=1e-5)
        else:
            self.icm_optim = None

        self.shuffle_agent_ids()

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

        else:
            msg  = f"ERROR: unknown action dtype of {self.action_dtype} "
            msg += "encountered when getting tokened action block."
            rank_print(msg)
            comm.Abort()

        return action_block

    def _multi_discrete_prob_to_one_hot(self, actions):
        """
        Convert a batch of multi-discrete action probabilities into a
        multi-discrete one-hot vector.
        NOTE: multi-discrete one-hot vectors are not really one-hot vectors
        since they have multiple 1s, but they can be decomposed into their
        invidiual parts which are true one-hots.

        Parameters:
        -----------
        actions: torch tensor
            A batch of actions having shape (batch_size, num_agents,
            action_pred_size).

        Returns:
        --------
        torch tensor:
            A multi-discrete one-hot tensor having the same shape
            as the input actions.
        """
        batch_size = actions.shape[0]
        num_agents = actions.shape[1]
    
        one_hot_actions = torch.zeros((batch_size, num_agents, self.action_pred_size))
        start = 0
        for a_idx, dim in enumerate(self.action_space.nvec):
            stop = start + dim
            one_hot_actions[:, :, start : stop] = \
                t_func.one_hot(actions[:, :, a_idx], dim)

        return one_hot_actions

    def _evaluate_actions(self, obs, batch_actions):
        """
        Evaluate a batch of actions given their associated observations.

        Parameters:
        -----------
        obs: tensor
            A batch of agent observations having shape
            (batch_size, num_agents, obs_size).
        batch_actions: tensor
            A batch of actions having shape
            (batch_size, num_agents, action_size).

        Returns:
        --------
        tuple:
            (values, log(action probabilities), entropy)
        """
        batch_size = obs.shape[0]
        num_agents = len(self.agent_ids)

        action_block = self._get_tokened_action_block(batch_size)

        #
        # We again shift the actions to include the start token here. This
        # will tell the critic to produce the next action probabilities
        # given the previous actions (the network uses a mask).
        #
        if self.action_dtype == "discrete":
            action_block[:, 1:, 1:] = t_func.one_hot(batch_actions,
                num_classes=self.action_pred_size)[:, :-1, :]

        elif self.action_dtype == "multi-discrete":
            one_hot_actions = self._multi_discrete_prob_to_one_hot(
                batch_actions)

            action_block[:, 1:, 1:] = one_hot_actions[:, :-1, :]

        else:
            action_block[:, 1:, :] = batch_actions[:, :-1, :]

        values, action_pred = self.actor_critic(obs, action_block)

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

        log_probs = log_probs.reshape((batch_size, num_agents, -1))
        entropy   = entropy.reshape((batch_size, num_agents, -1))

        return values, log_probs, entropy

    def _get_autoregressive_actions(self, encoded_obs):
        """
        Generate actions with exploration auto-regressively. This method
        starts by feeding an array containing a start token into the
        actor. The actor then produces the next action, which we add into
        the action block and feed it back into the actor to again get the
        next action, and this process continues untill all agents have been
        given actions.

        Parameters:
        -----------
        encoded_obs: tensor
            The agent observations that have been embedded/encoded by
            the critic.

        Returns:
        --------
        tuple:
            (output_action, output_raw_action, output_log_prob)
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

        output_log_prob = torch.zeros((batch_size, num_agents, 1),
            dtype=torch.float32)

        with torch.no_grad():
            for i in range(num_agents):
                action_pred = self.actor(action_block, encoded_obs)[:, i, :]
                dist        = self.actor.distribution.get_distribution(action_pred)

                action, raw_action = self.actor.distribution.sample_distribution(dist)
                log_prob = self.actor.distribution.get_log_probs(dist, raw_action)

                if self.action_dtype == "discrete":
                    output_action[:, i, :]     = action.unsqueeze(-1)
                    output_raw_action[:, i, :] = raw_action.unsqueeze(-1)
                    output_log_prob[:, i, :]   = log_prob.unsqueeze(-1)
                    action = t_func.one_hot(action, num_classes=self.action_pred_size)

                elif self.action_dtype == "multi-discrete":
                    action     = action.flatten()
                    raw_action = raw_action.flatten()
                    action     = action.reshape((self.envs_per_proc, 1, -1))
                    raw_action = raw_action.reshape((self.envs_per_proc, 1, -1))

                    output_action[:, i, :]     = action[:, 0, :]
                    output_raw_action[:, i, :] = raw_action[:, 0, :]
                    output_log_prob[:, i, :]   = log_prob.unsqueeze(-1)

                    one_hot_action = \
                        self._multi_discrete_prob_to_one_hot(action)

                    action = one_hot_action[:, 0, :]

                else:
                    log_prob                   = log_prob.unsqueeze(-1)
                    output_action[:, i, :]     = action
                    output_raw_action[:, i, :] = raw_action
                    output_log_prob[:, i, :]   = log_prob

                if i + 1 < num_agents:
                    action_block[:, i + 1, action_offset:] = action

        return output_action, output_raw_action, output_log_prob

    def _get_autoregressive_actions_without_exploration(self, encoded_obs):
        """
        Generate actions without exploration auto-regressively. This method
        starts by feeding an array containing a start token into the
        actor. The actor then produces the next action, which we add into
        the action block and feed it back into the actor to again get the
        next action, and this process continues untill all agents have been
        given actions.

        Parameters:
        -----------
        encoded_obs: tensor
            The agent observations that have been embedded/encoded by
            the critic.

        Returns:
        --------
        torch tensor:
            output_action
        """
        batch_size   = encoded_obs.shape[0]
        num_agents   = len(self.agent_ids)
        action_block = self._get_tokened_action_block(batch_size)

        if self.action_dtype in ["discrete", "multi-discrete"]:
            action_offset = 1
            output_action = torch.zeros((batch_size,
                num_agents, self.action_dim)).long()

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
                    action = action.reshape((self.envs_per_proc, 1, -1))

                    output_action[:, i, :] = action

                    one_hot_action = \
                        self._multi_discrete_prob_to_one_hot(action)

                    action = one_hot_action[:, 0, :]

                else:
                    output_action[:, i, :] = action

                if i + 1 < num_agents:
                    action_block[:, i + 1, action_offset:] = action

        return output_action

    def get_rollout_actions(self, obs):
        """
        Get actions for an ongoing rollout.

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
            msg  = "ERROR: _get_action expects a "
            msg ++ "batch of observations but "
            msg += "instead received shape {}.".format(obs.shape)
            rank_print(msg)
            comm.Abort()

        #
        # The incoming shape is (num_agents, num_envs, obs_size). MAT wants
        # (num_envs, num_agents, obs_size).
        #
        obs = np.swapaxes(obs, 0, 1)
        t_obs = torch.tensor(obs, dtype=torch.float32)
        t_obs = t_obs.to(self.device)

        encoded_obs, _ = self.critic(t_obs)
        actions, raw_actions, log_probs = \
            self._get_autoregressive_actions(encoded_obs)

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
        Update the weights of our actor/critic class.

        Parameters:
        -----------
        actor_loss: torch tensor
            The total loss for our actor.
        critic_loss: torch tensor
            The total loss for our critic.
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
        dict:
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
        if len(obs.shape) < 3:
            msg  = "ERROR: _get_actions expects a "
            msg += "batch of agent observations but "
            msg += "instead received shape {}.".format(obs.shape)
            rank_print(msg)
            comm.Abort()

        #
        # The incoming shape is (num_agents, num_envs, obs_size). MAT wants
        # (num_envs, num_agents, obs_size).
        #
        obs = np.swapaxes(obs, 0, 1)
        t_obs = torch.tensor(obs, dtype=torch.float32)
        t_obs = t_obs.to(self.device)

        encoded_obs, _ = self.critic(t_obs)
        actions, _, _ = \
            self._get_autoregressive_actions(encoded_obs)

        actions = torch.swapaxes(actions, 0, 1)

        return actions

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
        t_obs = torch.tensor(obs, dtype=torch.float32)
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
        Apply any constraints needed when resetting the environment.

        NOTE: This may alter the values returned by the environment.

        Parameters:
        -----------
        obs: dict
            Dictionary mapping agent ids to actor observations.
        critic_obs: dict
            Dictionary mapping agent ids to critic observations.
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

    def _save_policies(self, save_path):
        """
        Save our policies.

        Parameters:
        -----------
        save_path: str
            The state path to save the policy to.
        """
        self.actor_critic.save(save_path)

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
        try:
            self.actor_critic.load(load_path)

            if self.enable_icm:
                self.icm_model.load(load_path)

        except Exception as e:
            if tag != "latest":
                raise Exception(e)

            #
            # Backward compatibility. Support old saves that don't have 
            # the "latest" dir.
            #
            policy_dir = "{}-policy".format(self.name)
            load_path = os.path.join(load_path, policy_dir)
            self.actor_critic.load(load_path)

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

        optim_f = os.path.join(save_path, f"actor_critic_optim_{rank}")
        torch.save(self.actor_critic_optim.state_dict(), optim_f)

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

            optim_f = os.path.join(load_path, f"actor_critic_optim_{load_rank}")
            if not os.path.exists(optim_f):
                optim_f = os.path.join(load_path, f"actor_critic_optim_0")

            self.actor_critic_optim.load_state_dict(torch.load(optim_f))

            if self.enable_icm:
                icm_optim_f = os.path.join(load_path, f"icm_optim_{load_rank}")
                if not os.path.exists(icm_optim_f):
                    icm_optim_f = os.path.join(load_path, f"icm_optim_0")

                self.icm_optim.load_state_dict(torch.load(icm_optim_f))
        except Exception:
            rank_print("WARNING: unable to find saved optimizers to load. Skipping...")

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

    def get_agent_shared_intrinsic_rewards(self,
                                           prev_obs,
                                           obs,
                                           actions):
        """
        Query the ICM for an agent-grouped intrinsic reward.

        Parameters:
        -----------
        prev_obs: dict
            A dictionary mapping agent ids to previous observations.
        obs: dict
            A dictionary mapping agent ids to current observations.
        actions: dict
            A dictionary mapping agent ids to actions taken.
        """
        batch_size           = obs[tuple(obs.keys())[0]].shape[0]
        shared_obs_shape     = (batch_size,) + (self.num_agents,) + self.actor_obs_space.shape
        shared_actions_shape = (batch_size,) + (self.num_agents,) + self.action_space.shape

        shared_prev_obs = np.zeros(shared_obs_shape, dtype=np.float32)
        shared_obs      = np.zeros(shared_obs_shape, dtype=np.float32)
        shared_actions  = np.zeros(shared_actions_shape, dtype=self.action_space.dtype)

        for agent_idx, agent_id in enumerate(self.icm_agent_ids):

            agent_prev_obs = prev_obs[agent_id]
            agent_obs      = obs[agent_id]
            agent_action   = actions[agent_id]

            if len(agent_obs.shape) < 2:
                msg  = "ERROR: get_agent_shared_intrinsic_reward expects a batch of "
                msg += "observations but "
                msg += "instead received shape {}.".format(agent_obs.shape)
                rank_print(msg)
                comm.Abort()

            shared_prev_obs[:, agent_idx] = agent_prev_obs
            shared_obs[:, agent_idx]      = agent_obs
            shared_actions[:, agent_idx]  = agent_action

        #
        # Now we reshape into the format that ICM expects.
        #
        shared_obs_shape     = (batch_size,) + self.icm_model.obs_space.shape
        shared_actions_shape = (batch_size,) + self.icm_model.action_space.shape

        shared_prev_obs = shared_prev_obs.reshape(shared_obs_shape)
        shared_obs      = shared_obs.reshape(shared_obs_shape)
        shared_actions  = shared_actions.reshape(shared_actions_shape)

        obs_1 = torch.tensor(shared_prev_obs,
            dtype=torch.float32).to(self.device)
        obs_2 = torch.tensor(shared_obs,
            dtype=torch.float32).to(self.device)

        if self.action_dtype in ["discrete", "multi-discrete"]:
            shared_actions = torch.tensor(shared_actions,
                dtype=torch.int64).to(self.device)

        elif self.action_dtype in ["continuous", "multi-binary"]:
            shared_actions = torch.tensor(shared_actions,
                dtype=torch.float32).to(self.device)

        if len(shared_actions.shape) != 2:
            shared_actions = shared_actions.unsqueeze(1)

        with torch.no_grad():
            intr_reward, _, _ = self.icm_model(obs_1, obs_2, shared_actions)

        intr_reward  = intr_reward.detach().cpu().numpy()
        intr_reward  = intr_reward.reshape((batch_size, -1))
        intr_reward *= self.intr_reward_weight()

        return intr_reward

import numpy as np
import gc
from torch.utils.data import Dataset
import sys
import torch
from ppo_and_friends.utils.mpi_utils import rank_print
from mpi4py import MPI

comm      = MPI.COMM_WORLD
rank      = comm.Get_rank()
num_procs = comm.Get_size()

class EpisodeInfo(object):

    def __init__(self,
                 starting_ts    = 0,
                 use_gae        = False,
                 gamma          = 0.99,
                 lambd          = 0.95,
                 bootstrap_clip = (-10., 10.)):
        """
            A container for storing episode information.

            Arguments:
                starting_ts    The timestep that this episode starts at.
                use_gae        Should we use the Generalized Advantage
                               Estimation algorithm when calculating advantages?
                gamma          The discount factor to apply when calculating
                               discounted sums. This is used for advantages and
                               "rewards-to-go" (expected discounted returns).
                labmd          A "smoothing" factor used in GAE.
                bootstrap_clip A value to clip our bootstrapped rewards to.
        """

        self.starting_ts              = starting_ts
        self.ending_ts                = -1
        self.use_gae                  = use_gae
        self.gamma                    = gamma
        self.lambd                    = lambd
        self.bootstrap_clip           = bootstrap_clip
        self.critic_observations      = []
        self.observations             = []
        self.next_observations        = []
        self.actions                  = []
        self.raw_actions              = []
        self.log_probs                = []
        self.rewards                  = []
        self.values                   = []
        self.actor_hidden             = []
        self.critic_hidden            = []
        self.actor_cell               = []
        self.critic_cell              = []
        self.rewards_to_go            = None
        self.advantages               = None
        self.length                   = 0
        self.is_finished              = False
        self.has_hidden_states        = False

    def compute_discounted_sums(self, array, gamma):
        """
            Compute the discounted sums from a given array,
            which is assumed to be in temmporal order,
            [t0, t1, t2, ..., tn], where t0 is a 'value' at
            time 0, and tn is a 'value' at time n. Note that value
            here is not to be confused with the value from a value
            function; it's just some number. It could be a reward,
            a value from a value function, or whatever else you'd like.

            The discounted value at time t, DVt, follows the recursive formula
                DVt = DVt + (gamma * DVt+1)
            Such that all future values are considered in the current DV but
            at a discount.

            That is,
                DSn     = tn
                DS(n-1) = t(n-1) + (gamma * tn)
                ...
                DS0     = t0 + (gamma * t1) + (gamma^2 * t2) + ...
                          + (gamma^n + tn)

            Arguments:
                array    The array to calculate a discounted sum for.

            Returns:
                A numpy array containing the discounted sums.
        """
        cumulative_array = np.zeros(len(array))
        last_idx         = len(array) - 1
        d_sum = 0

        for idx, val in enumerate(array[::-1]):
            d_sum = val + gamma * d_sum
            cumulative_array[last_idx - idx] = d_sum

        return cumulative_array

    def _compute_gae_advantages(self,
                                padded_values,
                                rewards):
        """
            Compute the General Advantage Estimates. This follows the
            general GAE equation.

            Arguments:
                padded_values    A list of values from this epsiode with one
                                 extra value added to the end. This will either
                                 be a 0 (if the episode finished) or a repeat
                                 of the last value.

            Returns:
                An array containing the GAEs.
        """
        deltas = rewards + (self.gamma * padded_values[1:]) - \
            padded_values[:-1]

        sum_gamma = self.gamma * self.lambd
        return self.compute_discounted_sums(deltas.tolist(), sum_gamma)

    def _compute_standard_advantages(self):
        """
            Use a standard method for computing advantages.
            Typically, we use Q - values.

        """
        advantages = self.rewards_to_go - self.values
        return advantages

    def add_info(self,
                 observation,
                 next_observation,
                 raw_action,
                 action,
                 value,
                 log_prob,
                 reward,
                 critic_observation = np.empty(0),
                 actor_hidden       = np.empty(0),
                 actor_cell         = np.empty(0),
                 critic_hidden      = np.empty(0),
                 critic_cell        = np.empty(0)):
        """
            Add info from a single step in an episode. These should be
            added consecutively, in the order they are encountered.

            Arguments:
                observation              The observation eliciting our action.
                next_observation         The observation resulting from our
                                         action.
                raw_action               The un-altered action (there are times
                                         when we squash our actions into a new
                                         range. This value is pre-squash).
                action                   The action taken at this step.
                value                    The predicted value at this step (from
                                         the critic).
                log_prob                 The log probability calculated at this
                                         step.
                reward                   The reward received at this step.
                critic_observation       The critic observation used in multi-
                                         agent environments eliciting our
                                         action.
                actor_hidden             The hidden state of the actor iff the
                                         actor is an lstm.
                actor_cell               The cell state of the actor iff the
                                         actor is an lstm.
                critic_hidden            The hidden state of the critic iff the
                                         critic is an lstm.
                critic_cell              The cell state of the critic iff the
                                         critic is an lstm.
        """

        if type(raw_action) == np.ndarray and len(raw_action.shape) > 1:
            raw_action = raw_action.squeeze()

        if type(action) == np.ndarray and len(action.shape) > 1:
            action = action.squeeze()

        self.observations.append(observation)
        self.next_observations.append(next_observation)
        self.actions.append(action)
        self.raw_actions.append(raw_action)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.critic_observations.append(critic_observation)

        ac_hidden_check = np.array(
            (len(actor_hidden),
             len(actor_cell),
             len(critic_hidden),
             len(critic_cell))).astype(np.int32)

        if ( (ac_hidden_check > 0).any() and
             (ac_hidden_check == 0).any() ):

            msg  = "ERROR: if hidden state is provided for either the actor "
            msg += "or the critic, both most be provided, but we received "
            msg += "only one. Bailing..."
            rank_print(msg)
            comm.Abort()

        if len(actor_hidden) > 0:

            self.has_hidden_states = True

            self.actor_hidden.append(
                actor_hidden.detach().cpu().numpy())

            self.critic_hidden.append(
                critic_hidden.detach().cpu().numpy())

            self.actor_cell.append(
                actor_cell.detach().cpu().numpy())

            self.critic_cell.append(
                critic_cell.detach().cpu().numpy())

    def compute_advantages(self):
        """
            Compute our advantages using either a standard formula or GAE.
        """
        #
        # TODO: we should have an option to pass in a container to
        # put advantages into. This would save on some space.
        #
        if self.use_gae:
            padded_values = np.concatenate((self.values, (self.ending_value,)))
            padded_values = padded_values.astype(np.float32)

            self.advantages = self._compute_gae_advantages(
                padded_values,
                self.rewards)
        else:
            self.advantages = self._compute_standard_advantages()

    def end_episode(self,
                    ending_ts,
                    terminal,
                    ending_value,
                    ending_reward):
        """
            End the episode.

            Arguments:
                ending_ts      The time step we ended on.
                terminal       Did we end in a terminal state?
                ending_value   The ending value of the episode.
                ending_reward  The ending reward of the episode.
        """
        self.ending_ts    = ending_ts
        self.terminal     = terminal
        self.length       = self.ending_ts - self.starting_ts
        self.is_finished  = True
        self.ending_value = ending_value

        #
        # Clipping the ending value can have dramaticly positive
        # effects on training. MountainCarContinuous is a great
        # example of an environment that I've seen struggle quite
        # a bit without a propper bs clip.
        #
        ending_reward = np.clip(
            ending_reward,
            self.bootstrap_clip[0],
            self.bootstrap_clip[1])

        padded_rewards = np.array(self.rewards + [ending_reward],
            dtype=np.float32)

        self.rewards_to_go = self.compute_discounted_sums(
            padded_rewards,
            self.gamma)[:-1]

        self.values = np.array(self.values).astype(np.float32)

        self.compute_advantages()


class PPODataset(Dataset):

    def __init__(self,
                 device,
                 action_dtype,
                 sequence_length = 1):
        """
            A PyTorch Dataset representing our rollout data.

            Arguments:
                device           The device we're training on.
                action_dtype     The action data dtype (discrete/continuous).
                sequence_length  If set to > 1, our dataset will return
                                 obervations as sequences of this length.
        """

        self.action_dtype         = action_dtype
        self.device               = device
        self.episodes             = []
        self.is_built             = False
        self.build_hidden_states  = False
        self.sequence_length      = sequence_length
        self.build_terminal_mask  = False

        if self.sequence_length <= 0:
            msg  = "ERROR: PPODataset must have a sequence length >= 1 "
            msg += "but received {}".format(self.sequence_length)
            rank_print(msg)
            comm.Abort()

        elif self.sequence_length > 1:
            self.build_terminal_mask = True

        self.actions                   = None
        self.raw_actions               = None
        self.critic_observations       = None
        self.observations              = None
        self.next_observations         = None
        self.rewards_to_go             = None
        self.log_probs                 = None
        self.ep_lens                   = None
        self.advantages                = None
        self.actor_hidden              = None
        self.critic_hidden             = None
        self.actor_cell                = None
        self.critic_cell               = None
        self.terminal_mask             = None
        self.values                    = None

    def add_episode(self, episode):
        """
            Add an episode to our dataset.

            Arguments:
                episode    The episode to add.
        """
        if episode.has_hidden_states:
            self.build_hidden_states = True

        elif self.build_hidden_states  and not episode.has_hidden_states:
            msg  = "ERROR: some episodes have hidden states while others "
            msg += "do not. Bailing..."
            rank_print(msg)
            comm.Abort()

        self.episodes.append(episode)

    def recalculate_advantages(self):
        """
            Recalculate our advantages. This can be used to mitigate using
            stale advantages when training over > 1 epochs.
        """
        if not self.is_built:
            msg  = "WARNING: recalculate_advantages was called before "
            msg += "the dataset has been built. Ignoring call."
            rank_print(msg)
            return

        val_idx = 0
        for ep in self.episodes:
            for ep_ts in range(ep.length):
                ep.values[ep_ts] = self.values[val_idx]
                val_idx += 1

            ep.compute_advantages()

    def build(self):
        """
            Build our dataset from the existing episodes.
        """

        if self.is_built:
            msg  = "ERROR: attempting to build a batch, but it's "
            msg += "already been built! Bailing..."
            rank_print(msg)
            comm.Abort()

        #
        # TODO: let's use numpy arrays to save on space.
        #
        self.actions                  = []
        self.raw_actions              = []
        self.critic_observations      = []
        self.observations             = []
        self.next_observations        = []
        self.rewards_to_go            = []
        self.log_probs                = []
        self.ep_lens                  = []
        self.advantages               = []
        self.actor_hidden             = []
        self.critic_hidden            = []
        self.actor_cell               = []
        self.critic_cell              = []
        self.values                   = []

        self.num_episodes             = len(self.episodes)
        self.total_timestates         = 0

        for ep in self.episodes:
            self.total_timestates += ep.length

        if self.build_terminal_mask:
            terminal_mask = np.zeros(self.total_timestates).astype(np.bool)
            cur_ts = 0

        for ep in self.episodes:

            if not ep.is_finished:
                msg  = "ERROR: attempting to build a batch using "
                msg += "an incomplete episode! Bailing..."
                rank_print(msg)
                comm.Abort()

            self.actions.extend(ep.actions)
            self.raw_actions.extend(ep.raw_actions)
            self.observations.extend(ep.observations)
            self.next_observations.extend(ep.next_observations)
            self.rewards_to_go.extend(ep.rewards_to_go)
            self.log_probs.extend(ep.log_probs)
            self.ep_lens.append(ep.length)
            self.advantages.extend(ep.advantages)
            self.values.extend(ep.values)
            self.critic_observations.extend(ep.critic_observations)

            if self.build_hidden_states:
                self.actor_hidden.extend(ep.actor_hidden)
                self.critic_hidden.extend(ep.critic_hidden)

                self.actor_cell.extend(ep.actor_cell)
                self.critic_cell.extend(ep.critic_cell)

            if self.build_terminal_mask:
                cur_ts += ep.length

                if ep.terminal:
                    terminal_mask[cur_ts - 1] = True

        if self.build_terminal_mask:
            max_ts = self.total_timestates - (self.sequence_length - 1)
            self.terminal_sequence_masks = np.array(
                [None] * max_ts, dtype=object)

            for ts in range(max_ts):
                mask     = np.zeros(self.sequence_length).astype(np.bool)
                mask_idx = 0
                stop_ts  = ts + self.sequence_length

                for mask_ts in range(ts, stop_ts):
                    if terminal_mask[mask_ts]: 
                        mask[mask_idx + 1:] = True
                        break

                    mask_idx += 1

                self.terminal_sequence_masks[ts] = mask

        if self.total_timestates != len(self.observations):
            error_msg  = "ERROR: expected the total timestates to match "
            error_msg += "the total number of observations, but got "
            error_msg += "{} vs {}".format(self.total_timestates,
                len(self.observations))
            rank_print(error_msg)
            comm.Abort()

        #
        # Note that log_probs is a list of tensors, so we'll skip converting
        # it to a numpy array.
        #
        self.actions                  = np.array(self.actions)
        self.raw_actions              = np.array(self.raw_actions)
        self.critic_observations      = np.array(self.critic_observations)
        self.observations             = np.array(self.observations)
        self.next_observations        = np.array(self.next_observations)
        self.rewards_to_go            = np.array(self.rewards_to_go)
        self.ep_lens                  = np.array(self.ep_lens)
        self.advantages               = np.array(self.advantages)
        self.values                   = torch.tensor(self.values)

        self.values = self.values.to(self.device)

        if self.build_hidden_states:
            #
            # Torch expects the shape to be
            # (num_lstm_layers, batch_size, hidden_size), but our
            # data loader will concat on the first dimension. So, we
            # need to transpose so that the batch size comes first.
            #
            self.actor_hidden = torch.tensor(
                np.concatenate(self.actor_hidden, axis=1)).to(self.device)

            self.critic_hidden = torch.tensor(
                np.concatenate(self.critic_hidden, axis=1)).to(self.device)

            self.actor_cell = torch.tensor(
                np.concatenate(self.actor_cell, axis=1)).to(self.device)

            self.critic_cell = torch.tensor(
                np.concatenate(self.critic_cell, axis=1)).to(self.device)

            self.actor_hidden  = torch.transpose(self.actor_hidden, 0, 1)
            self.actor_cell    = torch.transpose(self.actor_cell, 0, 1)
            self.critic_hidden = torch.transpose(self.critic_hidden, 0, 1)
            self.critic_cell   = torch.transpose(self.critic_cell, 0, 1)

        else:
            empty_state = np.zeros(self.total_timestates).astype(np.uint8)

            self.actor_hidden  = empty_state
            self.critic_hidden = empty_state

            self.actor_cell    = empty_state
            self.critic_cell   = empty_state

        self.advantages = torch.tensor(self.advantages,
            dtype=torch.float).to(self.device)

        self.observations = torch.tensor(self.observations,
            dtype=torch.float).to(self.device)

        self.next_observations = torch.tensor(self.next_observations,
            dtype=torch.float).to(self.device)

        self.critic_observations = torch.tensor(self.critic_observations,
            dtype=torch.float).to(self.device)

        self.log_probs = torch.tensor(self.log_probs,
            dtype=torch.float).to(self.device)

        self.rewards_to_go = torch.tensor(self.rewards_to_go,
            dtype=torch.float).to(self.device)

        if self.action_dtype in ["continuous", "multi-binary"]:
            self.actions = torch.tensor(self.actions,
                dtype=torch.float).to(self.device)

            self.raw_actions = torch.tensor(self.raw_actions,
                dtype=torch.float).to(self.device)

        elif self.action_dtype in ["discrete", "multi-discrete"]:
            self.actions = torch.tensor(self.actions,
                dtype=torch.long).to(self.device)
            self.raw_actions = torch.tensor(self.raw_actions,
                dtype=torch.long).to(self.device)

        self.is_built = True

    def __len__(self):
        """
            Get the length of our dataset.
        """
        return self.total_timestates - (self.sequence_length - 1)

    def __getitem__(self, idx):
        """
            Get data of length self.sequence_length starting at index
            idx.

            Arguments:
                idx    The starting point of the data to retrieve.

            Returns:
                All data associated with the given index in our dataset.
        """
        #
        # First, handle the easy case of using single time states.
        #
        if self.sequence_length == 1:
            return (self.critic_observations[idx],
                    self.observations[idx],
                    self.next_observations[idx],
                    self.raw_actions[idx],
                    self.actions[idx],
                    self.advantages[idx],
                    self.log_probs[idx],
                    self.rewards_to_go[idx],
                    self.actor_hidden[idx],
                    self.critic_hidden[idx],
                    self.actor_cell[idx],
                    self.critic_cell[idx],
                    idx)

        #
        # We want our time sequence to start at the most recent state,
        # which means we'll be building a "tail" of history samples.
        #
        idx  += (self.sequence_length - 1)
        start = idx - (self.sequence_length - 1)
        stop  = idx + 1

        glob_obs_seq = self.critic_observations[start : stop].clone()
        obs_seq      = self.observations[start : stop].clone()
        nxt_obs_seq  = self.next_observations[start : stop].clone()

        #
        # Once we hit a terminal state, all subsequent observations
        # need to be zero'd out so that the model can learn that future
        # observations don't exist in this trajectory.
        #
        term_mask              = self.terminal_sequence_masks[start]
        obs_seq[term_mask]     = 0.0
        nxt_obs_seq[term_mask] = 0.0

        return (glob_obs_seq,
                obs_seq,
                nxt_obs_seq,
                self.raw_actions[idx],
                self.actions[idx],
                self.advantages[idx],
                self.log_probs[idx],
                self.rewards_to_go[idx],
                self.actor_hidden[idx],
                self.critic_hidden[idx],
                self.actor_cell[idx],
                self.critic_cell[idx],
                idx)

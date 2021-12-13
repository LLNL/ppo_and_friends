import numpy as np
from torch.utils.data import Dataset
import torch

class EpisodeInfo(object):

    def __init__(self,
                 use_gae      = False,
                 gamma        = 0.99,
                 lambd        = 0.95):
        """
            A container for storing episode information.

            Arguments:
                use_gae       Should we use the Generalized Advantage
                              Estimation algorithm when calculating advantages?
                gamma         The discount factor to apply when calculating
                              discounted sums. This is used for advantages and
                              "rewards-to-go" (expected discounted returns).
                labmd         A "smoothing" factor used in GAE.
        """

        self.use_gae           = use_gae
        self.gamma             = gamma
        self.lambd             = lambd
        self.observations      = []
        self.next_observations = []
        self.actions           = []
        self.log_probs         = []
        self.rewards           = []
        self.values            = []
        self.rewards_to_go     = None
        self.advantages        = None
        self.length            = 0
        self.is_finished       = False

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

    def _compute_gae_advantages(self, padded_values):
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

        deltas = self.rewards + (self.gamma * padded_values[1:]) - \
            padded_values[:-1]

        sum_gamma = self.gamma * self.lambd
        return self.compute_discounted_sums(deltas.tolist(), sum_gamma)

    def _compute_standard_advantages(self):
        """
            Use a standard method for computing advantages.
            Typically, we use Q - values, but we reverse this
            in practice (probably because we're using gradient
            ascent instead of descent, but this is not explicit;
            we just negate the loss).

        """
        advantages = self.rewards_to_go - self.values
        return advantages

    def add_info(self,
                 observation,
                 next_observation,
                 action,
                 value,
                 log_prob,
                 reward):

        self.observations.append(observation)
        self.next_observations.append(next_observation)
        self.actions.append(action)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)

    def end_episode(self,
                    ending_value,
                    episode_length):

        self.length      = episode_length
        self.is_finished = True

        self.rewards_to_go = self.compute_discounted_sums(self.rewards,
            self.gamma)

        if self.use_gae:
            padded_values  = np.array(self.values + [ending_value],
                dtype=np.float32)

            self.advantages = self._compute_gae_advantages(padded_values)

        else:
            self.advantages = self._compute_standard_advantages()


class PPODataset(Dataset):

    def __init__(self,
                 device,
                 action_type):

        self.action_type = action_type
        self.device      = device
        self.episodes    = []
        self.is_built    = False

        self.actions           = None
        self.observations      = None
        self.next_observations = None
        self.rewards_to_go     = None
        self.log_probs         = None
        self.ep_lens           = None
        self.advantages        = None

    def add_episode(self, episode):
        self.episodes.append(episode)

    def build(self):

        if self.is_built:
            msg  = "ERROR: attempting to build a batch, but it's "
            msg += "already been built! Bailing..."
            print(msg)
            sys.exit(1)

        self.actions           = []
        self.observations      = []
        self.next_observations = []
        self.rewards_to_go     = []
        self.log_probs         = []
        self.ep_lens           = []
        self.advantages        = []

        for ep in self.episodes:

            if not ep.is_finished:
                msg  = "ERROR: attempting to build a batch using "
                msg += "an incomplete episode! Bailing..."
                print(msg)
                sys.exit(1)

            self.actions.extend(ep.actions)
            self.observations.extend(ep.observations)
            self.next_observations.extend(ep.next_observations)
            self.rewards_to_go.extend(ep.rewards_to_go)
            self.log_probs.extend(ep.log_probs)
            self.ep_lens.append(ep.length)
            self.advantages.extend(ep.advantages)

        self.advantages = torch.tensor(self.advantages,
            dtype=torch.float).to(self.device)
        self.advantages = (self.advantages - self.advantages.mean()) / \
            (self.advantages.std() + 1e-10)

        self.observations = torch.tensor(self.observations,
            dtype=torch.float).to(self.device)

        self.next_observations = torch.tensor(self.next_observations,
            dtype=torch.float).to(self.device)

        self.log_probs = torch.tensor(self.log_probs,
            dtype=torch.float).to(self.device)

        self.rewards_to_go = torch.tensor(self.rewards_to_go,
            dtype=torch.float).to(self.device)

        if self.action_type == "continuous":
            self.actions = torch.tensor(self.actions,
                dtype=torch.float).to(self.device)

        elif self.action_type == "discrete":
            self.actions = torch.tensor(self.actions,
                dtype=torch.long).to(self.device)

        self.is_built = True

    def __len__(self):
        return len(self.observations)

    def __getitem__(self, idx):
        return (self.observations[idx],
                self.next_observations[idx],
                self.actions[idx],
                self.advantages[idx],
                self.log_probs[idx],
                self.rewards_to_go[idx])



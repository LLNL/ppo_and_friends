import numpy as np
from torch.utils.data import Dataset
import torch

class EpisodeInfo(object):

    def __init__(self,
                 use_gae      = False,
                 gamma        = 0.99,
                 lambd        = 0.95,
                 reward_scale = 1.0,
                 obs_scale    = 1.0):

        self.use_gae       = use_gae
        self.reward_scale  = reward_scale
        self.obs_scale     = obs_scale
        self.gamma         = gamma
        self.lambd         = lambd
        self.observations  = []
        self.actions       = []
        self.log_probs     = []
        self.rewards       = []
        self.values        = []
        self.rewards_to_go = None
        self.advantages    = None
        self.length        = 0
        self.is_finished   = False

    def compute_discounted_sum(self, array, gamma):
        cumulative_array = np.zeros(len(array))
        last_idx         = len(array) - 1
        d_sum = 0

        for idx, val in enumerate(array[::-1]):
            d_sum = val + gamma * d_sum
            cumulative_array[last_idx - idx] = d_sum

        return cumulative_array

    def _compute_gae_advantages(self, padded_values):

        deltas  = self.rewards + (self.gamma * padded_values[1:]) - padded_values[:-1]

        sum_gamma = self.gamma * self.lambd
        return self.compute_discounted_sum(deltas.tolist(), sum_gamma)

    def _compute_standard_advantages(self):
        advantages = self.rewards_to_go - self.values
        return advantages

    def add_info(self,
                 observation,
                 action,
                 value,
                 log_prob,
                 reward):

        self.observations.append(observation / self.obs_scale)
        self.actions.append(action)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.rewards.append(reward / self.reward_scale)

    def end_episode(self,
                    ending_value,
                    episode_length):

        self.length      = episode_length
        self.is_finished = True

        self.rewards_to_go = self.compute_discounted_sum(self.rewards,
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

        self.actions       = None
        self.observations  = None
        self.rewards_to_go = None
        self.log_probs     = None
        self.ep_lens       = None
        self.advantages    = None

    def add_episode(self, episode):
        self.episodes.append(episode)

    def build(self):

        if self.is_built:
            msg  = "ERROR: attempting to build a batch, but it's "
            msg += "already been built! Bailing..."
            print(msg)
            sys.exit(1)

        self.actions       = []
        self.observations  = []
        self.rewards_to_go = []
        self.log_probs     = []
        self.ep_lens       = []
        self.advantages    = []

        for ep in self.episodes:

            if not ep.is_finished:
                msg  = "ERROR: attempting to build a batch using "
                msg += "an incomplete episode! Bailing..."
                print(msg)
                sys.exit(1)

            self.actions.extend(ep.actions)
            self.observations.extend(ep.observations)
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
        self.log_probs = torch.tensor(self.log_probs,
            dtype=torch.float).to(self.device)
        self.rewards_to_go = torch.tensor(self.rewards_to_go,
            dtype=torch.float).to(self.device)

        if self.action_type == "continuous":
            self.actions = torch.tensor(self.actions,
                dtype=torch.float).to(self.device)
        elif self.action_type == "discrete":
            self.actions = torch.tensor(self.actions,
                dtype=torch.int32).to(self.device)

        self.is_built = True

    def __len__(self):
        return len(self.observations)

    def __getitem__(self, idx):
        return (self.observations[idx],
                self.actions[idx],
                self.advantages[idx],
                self.log_probs[idx],
                self.rewards_to_go[idx])



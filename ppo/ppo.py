from network import LinearNN, LinearNN2
import sys
import pickle
import numpy as np
import os
from torch.distributions import MultivariateNormal
from torch.distributions import Categorical
import torch
from torch.optim import Adam
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


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


class PPO(object):

    def __init__(self,
                 env,
                 device,
                 action_type,
                 use_gae,
                 reward_scale = 1.0,
                 obs_scale    = 1.0,
                 render       = False,
                 load_state   = False,
                 state_path   = "./"):

        if np.issubdtype(env.action_space.dtype, np.floating):
            self.act_dim = env.action_space.shape[0]
        elif np.issubdtype(env.action_space.dtype, np.integer):
            self.act_dim = env.action_space.n

        self.obs_dim      = env.observation_space.shape[0]
        self.env          = env
        self.device       = device
        self.state_path   = state_path
        self.render       = render
        self.action_type  = action_type
        self.use_gae      = use_gae
        self.reward_scale = reward_scale
        self.obs_scale    = obs_scale

        self.status_dict  = {}
        self.status_dict["iteration"] = 0
        self.status_dict["running score mean"] = 0
        self.status_dict["total episodes"] = 0

        need_softmax = False
        if action_type == "discrete":
            need_softmax = True

        self.actor  = LinearNN2(
            "actor", 
            self.obs_dim, 
            self.act_dim, 
            need_softmax)

        self.critic = LinearNN2(
            "critic", 
            self.obs_dim, 
            1,
            False)

        self.actor  = self.actor.to(device)
        self.critic = self.critic.to(device)

        if load_state:
            if not os.path.exists(state_path):
                msg  = "WARNING: state_path does not exist. Unable "
                msg += "to load state."
                print(msg)
            else:
                self.load()

        self._init_hyperparameters()

        self.cov_var = torch.full(size=(self.act_dim,), fill_value=0.5)
        self.cov_mat = torch.diag(self.cov_var)

        self.actor_optim  = Adam(self.actor.parameters(), lr=self.lr)
        self.critic_optim = Adam(self.critic.parameters(), lr=self.lr)

        if not os.path.exists(state_path):
            os.makedirs(state_path)

    def _init_hyperparameters(self):
        self.batch_size = 512
        self.timesteps_per_batch = 2048
        self.max_timesteps_per_episode = 200
        self.gamma = 0.99
        self.epochs_per_iteration = 10
        self.clip = 0.2
        self.action_std = 0.6
        self.action_std_decay_rate = 0.05
        self.min_action_std = 0.1
        self.action_decay_freq = 250000
        self.lr = 3e-4
        #self.lr = 0.00095

    def get_action(self, obs):

        if self.action_type == "continuous":
            t_obs = torch.tensor(obs, dtype=torch.float).to(self.device)
            mean_action = self.actor(t_obs).cpu().detach()

            dist     = MultivariateNormal(mean_action, self.cov_mat)
            action   = dist.sample()
            log_prob = dist.log_prob(action)

        elif self.action_type == "discrete":
            t_obs = torch.tensor(obs, dtype=torch.float).to(self.device)
            probs = self.actor(t_obs).cpu().detach()

            dist     = Categorical(probs)
            action   = dist.sample()
            log_prob = dist.log_prob(action)
            action   = action.int().unsqueeze(0)

        return action.detach().numpy(), log_prob.detach().to(self.device)

    def evaluate(self, batch_obs, batch_actions):
        values = self.critic(batch_obs).squeeze()

        if self.action_type == "continuous":
            mean = self.actor(batch_obs).cpu()
            dist = MultivariateNormal(mean, self.cov_mat)
            log_probs = dist.log_prob(batch_actions.unsqueeze(1).cpu())

        elif self.action_type == "discrete":
            batch_actions = batch_actions.flatten()
            probs         = self.actor(batch_obs).cpu()
            dist          = Categorical(probs)
            log_probs     = dist.log_prob(batch_actions.cpu())

        return values, log_probs.to(self.device), dist.entropy().to(self.device)

    def print_status(self):

        print("\n--------------------------------------------------------")
        print("Status Report:")
        for key in self.status_dict:
            print("    {}: {}".format(key, self.status_dict[key]))
        print("--------------------------------------------------------")

    def rollout(self):
        dataset        = PPODataset(self.device, self.action_type)
        total_episodes = 0  
        total_ts       = 0
        total_rewards  = 0

        while total_ts < self.timesteps_per_batch:
            episode_info = EpisodeInfo(
                use_gae      = self.use_gae,
                reward_scale = self.reward_scale)

            total_episodes  += 1
            done             = False
            obs              = self.env.reset()

            for ts in range(self.max_timesteps_per_episode):
                if self.render:
                    self.env.render()

                total_ts += 1

                if self.obs_scale != 1.0:
                    obs /= self.obs_scale

                action, log_prob = self.get_action(obs)

                t_obs    = torch.tensor(obs, dtype=torch.float).to(self.device)
                value    = self.critic(t_obs)
                prev_obs = obs.copy()

                if self.action_type == "discrete":
                    obs, reward, done, _ = self.env.step(action.squeeze())

                    episode_info.add_info(
                        prev_obs,
                        action.item(),
                        value.item(),
                        log_prob,
                        reward)

                else:
                    obs, reward, done, _ = self.env.step(action)

                    episode_info.add_info(
                        prev_obs,
                        action.item(),
                        value,
                        log_prob,
                        reward)

                total_rewards += reward

                if done:
                    episode_info.end_episode(0.0, ts + 1)
                    break

                elif ts == (self.max_timesteps_per_episode - 1):
                    episode_info.end_episode(value, ts + 1)

            dataset.add_episode(episode_info)

        self.status_dict["running score mean"] = total_rewards / total_episodes
        self.status_dict["total episodes"]     = total_episodes

        dataset.build()

        return dataset

    def learn(self, total_timesteps):

        ts = 0
        while ts < total_timesteps:
            dataset = self.rollout()

            ts += np.sum(dataset.ep_lens)
            self.status_dict["iteration"] += 1

            data_loader = DataLoader(
                dataset,
                batch_size = self.batch_size,
                shuffle    = False)#FIXME: turn on shuffling?

            for _ in range(self.epochs_per_iteration):
                self._batch_train(data_loader)

            self.print_status()
            self.save()

    def _batch_train(self, data_loader):
        for obs, actions, advantages, log_probs, rewards_tg in data_loader:
            values, curr_log_probs, entropy = self.evaluate(obs, actions)

            # new p / old p
            ratios = torch.exp(curr_log_probs - log_probs)
            surr1  = ratios * advantages
            surr2  = torch.clamp(ratios, 1 - self.clip, 1 + self.clip) * \
                advantages

            actor_loss  = (-torch.min(surr1, surr2)).mean() - 0.01 * entropy.mean()
            critic_loss = nn.MSELoss()(values, rewards_tg)

            self.actor_optim.zero_grad()
            actor_loss.backward(retain_graph=True)
            self.actor_optim.step()

            self.critic_optim.zero_grad()
            critic_loss.backward()
            self.critic_optim.step()

    def save(self):
        self.actor.save(self.state_path)
        self.critic.save(self.state_path)

        state_file = os.path.join(self.state_path, "state.pickle")
        with open(state_file, "wb") as out_f:
            pickle.dump(self.status_dict, out_f,
                protocol=pickle.HIGHEST_PROTOCOL)

    def load(self):
        self.actor.load(self.state_path)
        self.critic.load(self.state_path)

        state_file = os.path.join(self.state_path, "state.pickle")
        with open(state_file, "rb") as in_f:
            self.status_dict = pickle.load(in_f)

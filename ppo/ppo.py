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


class PPO(object):

    def __init__(self,
                 env,
                 device,
                 action_type,
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
        self.timesteps_per_batch = 2048
        self.max_timesteps_per_episode = 200
        self.gamma = 0.99
        self.epochs_per_iteration = 10
        self.clip = 0.2
        self.lr = 3e-4

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

    def compute_rewards_tg(self, batch_rewards):
        batch_rewards_tg = []

        for ep_rewards in reversed(batch_rewards):
            discounted_reward = 0

            for reward in reversed(ep_rewards):

                discounted_reward = reward + discounted_reward * self.gamma

                #FIXME: we can be a lot more effecient here.
                batch_rewards_tg.insert(0, discounted_reward)

        batch_rewards_tg = torch.tensor(batch_rewards_tg, dtype=torch.float).to(self.device)
        return batch_rewards_tg

    def evaluate(self, batch_obs, batch_actions):
        values = self.critic(batch_obs).squeeze()

        if self.action_type == "continuous":
            mean = self.actor(batch_obs).cpu()
            dist = MultivariateNormal(mean, self.cov_mat)
            log_probs = dist.log_prob(batch_actions.cpu())

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
        batch_obs        = [] # observations.
        batch_actions    = [] # actions.
        batch_log_probs  = [] # log probs of each action.
        batch_rewards    = [] # rewards.
        batch_rewards_tg = [] # rewards to go.
        batch_ep_lens    = [] # episode lengths.

        total_episodes = 0  
        total_ts       = 0
        total_rewards  = 0

        while total_ts < self.timesteps_per_batch:

            total_episodes  += 1
            ep_rewards       = []
            done             = False
            obs              = self.env.reset()

            for ts in range(self.max_timesteps_per_episode):
                if self.render:
                    self.env.render()

                total_ts += 1

                if self.obs_scale != 1.0:
                    obs /= self.obs_scale

                batch_obs.append(obs)
                action, log_prob = self.get_action(obs)

                #FIXME: can we make this cleaner?
                if self.action_type == "discrete":
                    obs, reward, done, _ = self.env.step(action.squeeze())
                else:
                    obs, reward, done, _ = self.env.step(action)

                total_rewards += reward
                ep_rewards.append(reward)
                batch_actions.append(action)
                batch_log_probs.append(log_prob)

                if done:
                    break

            batch_ep_lens.append(ts + 1)
            batch_rewards.append(ep_rewards)

        self.status_dict["running score mean"] = total_rewards / total_episodes
        self.status_dict["total episodes"]     = total_episodes

        batch_obs       = torch.tensor(batch_obs, dtype=torch.float).to(self.device)
        batch_log_probs = torch.tensor(batch_log_probs, dtype=torch.float).to(self.device)

        if self.action_type == "continuous":
            batch_actions = torch.tensor(batch_actions, dtype=torch.float).to(self.device)
        elif self.action_type == "discrete":
            batch_actions = torch.tensor(batch_actions, dtype=torch.int32).to(self.device)

        if self.reward_scale != 1.0:
            scaled_rewards = []
            for reward_list in batch_rewards:
                scaled_rewards.append(list(np.array(reward_list) /
                    self.reward_scale))

            batch_rewards = scaled_rewards

        batch_rewards_tg = self.compute_rewards_tg(batch_rewards).to(self.device)

        return batch_obs, batch_actions, batch_log_probs, batch_rewards_tg,\
            batch_ep_lens

    def get_advantages(self):
        pass

    def learn(self, total_timesteps):

        t_so_far = 0
        while t_so_far < total_timesteps:
            batch_obs, batch_actions, batch_log_probs, \
                batch_rewards_tg, batch_ep_lens = self.rollout()

            t_so_far += np.sum(batch_ep_lens)
            self.status_dict["iteration"] += 1

            values, _, _  = self.evaluate(batch_obs, batch_actions)
            advantages = batch_rewards_tg - values.detach()

            advantages = (advantages - advantages.mean()) / \
                (advantages.std() + 1e-10)

            for _ in range(self.epochs_per_iteration):
                values, curr_log_probs, entropy = self.evaluate(batch_obs, batch_actions)

                # new p / old p
                ratios = torch.exp(curr_log_probs - batch_log_probs)
                surr1  = ratios * advantages
                surr2  = torch.clamp(ratios, 1 - self.clip, 1 + self.clip) * advantages

                #FIXME: people seem to be using entropy here. Does it make a difference?
                actor_loss  = (-torch.min(surr1, surr2)).mean() - 0.01 * entropy.mean()
                #actor_loss  = (-torch.min(surr1, surr2)).mean()
                critic_loss = nn.MSELoss()(values, batch_rewards_tg)

                self.actor_optim.zero_grad()
                actor_loss.backward(retain_graph=True)
                self.actor_optim.step()

                self.critic_optim.zero_grad()
                critic_loss.backward()
                self.critic_optim.step()

            self.print_status()
            self.save()


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

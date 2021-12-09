from networks import LinearNN, LinearNN2, StateActionPredictor
import sys
import pickle
import numpy as np
import os
from torch.distributions import MultivariateNormal
from torch.distributions import Categorical
import torch
from torch.optim import Adam
from torch import nn
from torch.utils.data import DataLoader
from utils import EpisodeInfo, PPODataset


class PPO(object):

    def __init__(self,
                 env,
                 device,
                 action_type,
                 use_gae,
                 use_icm      = False,
                 icm_beta     = 0.8,
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
        self.use_icm      = use_icm
        self.icm_beta     = icm_beta
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

        if self.use_icm:
            self.icm_model = StateActionPredictor(
                self.obs_dim,
                self.act_dim,
                self.action_type)

            self.icm_model.to(device)
            self.status_dict["icm loss"] = 0

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

        if self.use_icm:
            self.icm_optim = Adam(self.icm_model.parameters(), lr=self.lr)

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

        t_obs = torch.tensor(obs, dtype=torch.float).to(self.device)

        with torch.no_grad():
            action_pred = self.actor(t_obs)

        if self.action_type == "continuous":
            action_mean = action_pred.cpu().detach()
            dist        = MultivariateNormal(action_mean, self.cov_mat)
            action      = dist.sample()
            log_prob    = dist.log_prob(action)

        elif self.action_type == "discrete":
            probs    = action_pred.cpu().detach()
            dist     = Categorical(probs)
            action   = dist.sample()
            log_prob = dist.log_prob(action)
            action   = action.int().unsqueeze(0)

        return action.detach().numpy(), log_prob.detach().to(self.device)

    def evaluate(self, batch_obs, batch_actions):
        values = self.critic(batch_obs).squeeze()

        if self.action_type == "continuous":
            action_mean = self.actor(batch_obs).cpu()
            dist        = MultivariateNormal(action_mean, self.cov_mat)
            log_probs   = dist.log_prob(batch_actions.unsqueeze(1).cpu())

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
                    ep_action = action.item()
                    ep_value  = value.item()

                elif self.action_type == "continuous":
                    obs, reward, done, _ = self.env.step(action)
                    ep_action = action.item()
                    ep_value  = value

                if self.use_icm:
                    obs_1 = torch.tensor(prev_obs,
                        dtype=torch.float).to(self.device).unsqueeze(0)
                    obs_2 = torch.tensor(obs,
                        dtype=torch.float).to(self.device).unsqueeze(0)

                    if self.action_type == "discrete":
                        action = torch.tensor(action,
                            dtype=torch.long).to(self.device).unsqueeze(0)
                    elif self.action_type == "continuous":
                        action = torch.tensor(action,
                            dtype=torch.float).to(self.device).unsqueeze(0)

                    with torch.no_grad():
                        intrinsic_reward, _, _ = self.icm_model(obs_1, obs_2, action)

                reward += intrinsic_reward.cpu().item()

                episode_info.add_info(
                    prev_obs,
                    obs,
                    ep_action,
                    ep_value,
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
                shuffle    = True)

            for _ in range(self.epochs_per_iteration):
                self._ppo_batch_train(data_loader)

            if self.use_icm:
                for _ in range(self.epochs_per_iteration):
                    self._icm_batch_train(data_loader)

            self.print_status()
            self.save()

    def _ppo_batch_train(self, data_loader):

        for obs, _, actions, advantages, log_probs, rewards_tg in data_loader:

            values, curr_log_probs, entropy = self.evaluate(obs, actions)

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

    def _icm_batch_train(self, data_loader):

        total_icm_loss = 0
        counter = 0

        for obs, next_obs, actions, _, _, _ in data_loader:

            actions = actions.unsqueeze(1)

            _, inv_loss, f_loss = self.icm_model(obs, next_obs, actions)

            icm_loss = (((1.0 - self.icm_beta) * f_loss) +
                (self.icm_beta * inv_loss))

            total_icm_loss += icm_loss.item()

            self.icm_optim.zero_grad()
            icm_loss.backward()
            self.icm_optim.step()

            counter += 1

        self.status_dict["icm loss"] = total_icm_loss / counter


    def save(self):
        self.actor.save(self.state_path)
        self.critic.save(self.state_path)

        if self.use_icm:
            self.icm_model.save(self.state_path)

        state_file = os.path.join(self.state_path, "state.pickle")
        with open(state_file, "wb") as out_f:
            pickle.dump(self.status_dict, out_f,
                protocol=pickle.HIGHEST_PROTOCOL)

    def load(self):
        self.actor.load(self.state_path)
        self.critic.load(self.state_path)

        if self.use_icm:
            self.icm_model.load(self.state_path)

        state_file = os.path.join(self.state_path, "state.pickle")
        with open(state_file, "rb") as in_f:
            self.status_dict = pickle.load(in_f)

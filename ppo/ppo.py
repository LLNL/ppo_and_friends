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
from utils import EpisodeInfo, PPODataset, get_action_type
from networks import ICM, LinearObservationEncoder


class PPO(object):

    def __init__(self,
                 env,
                 network,
                 device,
                 icm_network         = ICM,
                 icm_encoder         = LinearObservationEncoder,
                 lr                  = 3e-4,
                 min_lr              = 1e-4,
                 lr_dec              = 0.99,
                 lr_dec_freq         = 500,
                 max_ts_per_ep       = 200,
                 batch_size          = 256,
                 timesteps_per_batch = 2048,
                 gamma               = 0.99,
                 epochs_per_iter     = 10,
                 clip                = 0.2,
                 use_gae             = False,
                 use_icm             = False,
                 icm_beta            = 0.8,
                 ext_reward_scale    = 1.0,
                 intr_reward_scale   = 1.0,
                 entropy_weight      = 0.01,
                 target_kl           = 0.01,
                 render              = False,
                 load_state          = False,
                 state_path          = "./"):

        if np.issubdtype(env.action_space.dtype, np.floating):
            self.act_dim = env.action_space.shape[0]
        elif np.issubdtype(env.action_space.dtype, np.integer):
            self.act_dim = env.action_space.n

        action_type = get_action_type(env)

        if action_type == "unknown":
            print("ERROR: unknown action type!")
            sys.exit(1)

        #
        # Environments are very inconsistent! We need to check what shape
        # they expect actions to be in.
        #
        try:
            padded_shape = (1, self.act_dim)

            if action_type == "discrete":
                action = np.random.random(padded_shape)
            elif action_type == "continuous":
                action = np.random.randint(0, 1, padded_shape)

            env.reset()
            env.step(action)
            env.reset()
            self.need_action_squeeze = False
            
        except:
            action_shape = (self.act_dim,)

            if action_type == "discrete":
                action = np.random.random(action_shape)
            elif action_type == "continuous":
                action = np.random.randint(0, 1, action_shape)

            env.reset()
            env.step(action)
            env.reset()
            self.need_action_squeeze = True

        self.env                 = env
        self.device              = device
        self.state_path          = state_path
        self.render              = render
        self.action_type         = action_type
        self.use_gae             = use_gae
        self.use_icm             = use_icm
        self.icm_beta            = icm_beta
        self.ext_reward_scale    = ext_reward_scale
        self.intr_reward_scale   = intr_reward_scale
        self.lr                  = lr
        self.min_lr              = min_lr
        self.lr_dec              = lr_dec
        self.lr_dec_freq         = lr_dec_freq
        self.max_ts_per_ep       = max_ts_per_ep
        self.batch_size          = batch_size
        self.timesteps_per_batch = timesteps_per_batch
        self.gamma               = gamma
        self.target_kl           = target_kl
        self.epochs_per_iter     = epochs_per_iter
        self.clip                = clip
        self.entropy_weight      = entropy_weight
        self.obs_shape           = env.observation_space.shape

        #
        # Create a dictionary to track the status of training.
        #
        self.status_dict  = {}
        self.status_dict["iteration"]            = 0
        self.status_dict["longest run"]          = 0
        self.status_dict["running score mean"]   = 0
        self.status_dict["extrinsic score mean"] = 0
        self.status_dict["top score"]            = 0
        self.status_dict["total episodes"]       = 0
        self.status_dict["entropy"]              = 0
        self.status_dict["weighted entropy"]     = 0
        self.status_dict["actor loss"]           = 0
        self.status_dict["critic loss"]          = 0
        self.status_dict["kl mean"]              = 0
        self.status_dict["lr"]                   = self.lr

        need_softmax = False
        if action_type == "discrete":
            need_softmax = True

        use_conv2d_setup = False
        for base in network.__bases__:
            if base.__name__ == "PPOConv2dNetwork":
                use_conv2d_setup = True
                break

        if use_conv2d_setup:
            obs_dim = self.obs_shape

            self.actor = network(
                "actor", 
                obs_dim, 
                self.act_dim, 
                need_softmax)

            self.critic = network(
                "critic", 
                obs_dim, 
                1,
                False)

        else:
            obs_dim = self.obs_shape[0]
            self.actor = network(
                "actor", 
                obs_dim, 
                self.act_dim, 
                need_softmax)

            self.critic = network(
                "critic", 
                obs_dim, 
                1,
                False)

        self.actor  = self.actor.to(device)
        self.critic = self.critic.to(device)

        if self.use_icm:
            self.icm_model = icm_network(
                obs_dim     = obs_dim,
                act_dim     = self.act_dim,
                action_type = self.action_type,
                obs_encoder = icm_encoder)

            self.icm_model.to(device)
            self.status_dict["icm loss"] = 0
            self.status_dict["intrinsic score mean"] = 0

        if load_state:
            if not os.path.exists(state_path):
                msg  = "WARNING: state_path does not exist. Unable "
                msg += "to load state."
                print(msg)
            else:
                self.load()
                self.lr = min(self.status_dict["lr"], self.lr)
                self.status_dict["lr"] = self.lr

        self.cov_var = torch.full(size=(self.act_dim,), fill_value=0.5)
        self.cov_mat = torch.diag(self.cov_var)

        self.actor_optim  = Adam(self.actor.parameters(), lr=self.lr)
        self.critic_optim = Adam(self.critic.parameters(), lr=self.lr)

        if self.use_icm:
            self.icm_optim = Adam(self.icm_model.parameters(), lr=self.lr)

        if not os.path.exists(state_path):
            os.makedirs(state_path)

    def get_action(self, obs):

        t_obs = torch.tensor(obs, dtype=torch.float).to(self.device)
        t_obs = t_obs.unsqueeze(0)

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

        return action.detach().numpy(), log_prob.detach()

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
        dataset            = PPODataset(self.device, self.action_type)
        total_episodes     = 0  
        total_ts           = 0
        total_ext_rewards  = 0
        total_intr_rewards = 0
        top_ep_score       = 0
        total_score        = 0
        longest_run        = 0

        self.actor.eval()
        self.critic.eval()

        while total_ts < self.timesteps_per_batch:
            episode_info = EpisodeInfo(
                use_gae = self.use_gae)

            total_episodes  += 1
            done             = False
            obs              = self.env.reset()
            ep_score         = 0

            #
            # HACK: some environments are buggy and return inconsistent
            # observation shapes. We can enforce shapes here.
            #
            obs = obs.reshape(self.obs_shape)

            for ts in range(self.max_ts_per_ep):
                if self.render:
                    self.env.render()

                total_ts += 1
                action, log_prob = self.get_action(obs)


                t_obs    = torch.tensor(obs, dtype=torch.float).to(self.device)
                t_obs    = t_obs.unsqueeze(0)
                value    = self.critic(t_obs)
                prev_obs = obs.copy()

                if self.need_action_squeeze:
                    obs, ext_reward, done, _ = self.env.step(action.squeeze())
                else:
                    obs, ext_reward, done, _ = self.env.step(action)

                ep_action  = action.item()
                ep_value   = value.item()

                #
                # HACK: some environments are buggy and return inconsistent
                # observation shapes. We can enforce shapes here.
                #
                obs = obs.reshape(self.obs_shape)

                #
                # Some gym environments return arrays of single elements,
                # which is very annoying...
                #
                if type(ext_reward) == np.ndarray:
                    ext_reward = ext_reward[0]

                natural_reward = ext_reward
                ext_reward    *= self.ext_reward_scale

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
                            dtype=torch.float).to(self.device)

                    if len(action.shape) != 2:
                        action = action.unsqueeze(0)

                    with torch.no_grad():
                        intrinsic_reward, _, _ = self.icm_model(obs_1, obs_2, action)

                    intrinsic_reward  = intrinsic_reward.item()
                    intrinsic_reward *= self.intr_reward_scale

                    total_intr_rewards += intrinsic_reward

                    reward = ext_reward + intrinsic_reward

                else:
                    reward = ext_reward

                episode_info.add_info(
                    prev_obs,
                    obs,
                    ep_action,
                    ep_value,
                    log_prob,
                    reward)

                total_score       += reward
                ep_score          += natural_reward
                total_ext_rewards += natural_reward

                if done:
                    episode_info.end_episode(0, ts + 1)
                    break

                elif ts == (self.max_ts_per_ep - 1):
                    t_obs     = torch.tensor(obs, dtype=torch.float).to(self.device)
                    t_obs     = t_obs.unsqueeze(0)
                    nxt_value = self.critic(t_obs)
                    episode_info.end_episode(nxt_value.item(), ts + 1)

            dataset.add_episode(episode_info)
            top_ep_score = max(top_ep_score, ep_score)
            longest_run  = max(longest_run, ts)

        top_score         = max(top_ep_score, self.status_dict["top score"])
        running_ext_score = total_ext_rewards / total_episodes
        running_score     = total_score / total_episodes

        self.status_dict["running score mean"]   = running_score
        self.status_dict["extrinsic score mean"] = running_ext_score
        self.status_dict["top score"]            = top_score
        self.status_dict["total episodes"]       = total_episodes
        self.status_dict["longest run"]          = longest_run

        if self.use_icm:
            ism = total_intr_rewards / total_episodes
            self.status_dict["intrinsic score mean"] = ism

        dataset.build()

        return dataset

    def check_learning_rate(self):
        iteration = self.status_dict["iteration"]

        if (iteration != 0 and iteration % self.lr_dec_freq == 0
            and self.lr > self.min_lr):

            new_lr = max(self.lr * self.lr_dec, self.min_lr)

            msg  = "\n***Decreasing learning rate from "
            msg += "{} to {}***\n".format(self.lr, new_lr)
            print(msg)

            self.lr = new_lr
            self.status_dict["lr"] = self.lr

    def learn(self, total_timesteps):

        ts = 0
        while ts < total_timesteps:
            dataset = self.rollout()

            ts += np.sum(dataset.ep_lens)
            self.status_dict["iteration"] += 1

            self.check_learning_rate()

            data_loader = DataLoader(
                dataset,
                batch_size = self.batch_size,
                shuffle    = True)

            self.actor.train()
            self.critic.train()

            for _ in range(self.epochs_per_iter):

                self._ppo_batch_train(data_loader)

                if self.status_dict["kl mean"] > (1.5 * self.target_kl):
                    print("Target KL has been reached. Ending epoch early.")
                    break

            if self.use_icm:
                for _ in range(self.epochs_per_iter):
                    self._icm_batch_train(data_loader)

            self.print_status()
            self.save()

    def _ppo_batch_train(self, data_loader):

        total_actor_loss  = 0
        total_critic_loss = 0
        total_entropy     = 0
        total_w_entropy   = 0
        total_kl          = 0
        counter           = 0

        for obs, _, actions, advantages, log_probs, rewards_tg in data_loader:
            torch.cuda.empty_cache()

            if obs.shape[0] == 1 and (self.actor.uses_batch_norm or
                self.critic.uses_batch_norm):

                print("Skipping batch of size 1")
                print("    obs shape: {}".format(obs.shape))
                continue

            values, curr_log_probs, entropy = self.evaluate(obs, actions)

            #
            # We udpate our policy using gradient ascent. Our advantages relay
            # how much better or worse the outcome of various actions were than
            # "expected" (from our value approximator). Since actions that are
            # already very likely will be taken more often and thus updated
            # more often, we divide the gradient of the current probabilities
            # by the original probabilities. This helps lessen the impact of
            # frequent updates of probable actions while giving less probable
            # actions a chance to be considered.
            # We take the difference between the current and previous log probs
            # to further constrain updates and clip the output to a specified
            # range.
            #
            # FIXME: what's with the exponent?
            #
            ratios    = torch.exp(curr_log_probs - log_probs)
            surr1     = ratios * advantages
            surr2     = torch.clamp(
                ratios, 1 - self.clip, 1 + self.clip) * advantages
            total_kl += (log_probs - curr_log_probs).mean().item()

            #
            # We negate here to perform gradient ascent rather than descent.
            #
            actor_loss        = -(torch.min(surr1, surr2)).mean()
            total_actor_loss += actor_loss.item()
            total_entropy    += entropy.mean().item()
            actor_loss       -= self.entropy_weight * entropy.mean()

            if values.size() == torch.Size([]):
                values = values.unsqueeze(0)

            critic_loss        = nn.MSELoss()(values, rewards_tg)
            total_critic_loss += critic_loss.item()

            self.actor_optim.zero_grad()
            actor_loss.backward(retain_graph=True)
            self.actor_optim.step()

            self.critic_optim.zero_grad()
            critic_loss.backward()
            self.critic_optim.step()

            counter += 1

        w_entropy = total_entropy * self.entropy_weight

        self.status_dict["entropy"]          = total_entropy / counter
        self.status_dict["weighted entropy"] = w_entropy / counter
        self.status_dict["actor loss"]       = total_actor_loss / counter
        self.status_dict["critic loss"]      = total_critic_loss / counter
        self.status_dict["kl mean"]          = total_kl / counter

    def _icm_batch_train(self, data_loader):

        total_icm_loss = 0
        counter = 0

        for obs, next_obs, actions, _, _, _ in data_loader:
            torch.cuda.empty_cache()

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

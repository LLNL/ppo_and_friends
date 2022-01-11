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
from utils.episode_info import EpisodeInfo, PPODataset
from utils.misc import get_action_type, need_action_squeeze
from utils.decrementers import *
from networks import ICM, LinearObservationEncoder
import time


class PPO(object):

    #TODO: replace lr dec with a function to be applied.
    def __init__(self,
                 env,
                 ac_network,
                 device,
                 icm_network         = ICM,
                 icm_kw_args         = {},
                 ac_kw_args          = {},
                 lr                  = 3e-4,
                 min_lr              = 1e-4,
                 lr_dec              = None,
                 max_ts_per_ep       = 200,
                 batch_size          = 256,
                 ts_per_rollout      = 1024,
                 gamma               = 0.99,
                 lambd               = 0.95,
                 epochs_per_iter     = 10,
                 clip                = 0.2,
                 bootstrap_clip      = (-1.0, 1.0),
                 dynamic_bs_clip     = True,
                 use_gae             = True,
                 use_icm             = False,
                 icm_beta            = 0.8,
                 ext_reward_weight   = 1.0,
                 intr_reward_weight  = 1.0,
                 entropy_weight      = 0.01,
                 target_kl           = 100.0,
                 mean_window_size    = 100,
                 render              = False,
                 load_state          = False,
                 state_path          = "./",
                 save_best_only      = False):
        """
            Initialize the PPO trainer.

            Parameters:
                 env                  The environment to learn from.
                 ac_network           The actor/critic network.
                 device               A torch device to use for training.
                 icm_network          The network to use for ICM applications.
                 icm_kw_args          Extra keyword args for the ICM network.
                 ac_kw_args           Extra keyword args for the actor/critic
                                      networks.
                 lr                   The initial learning rate.
                 min_lr               The minimum learning rate.
                 lr_dec               A class that inherits from the Decrement
                                      class located in utils/decrementers.py.
                                      This class has a decrement function that
                                      will be used to updated the learning rate.
                 max_ts_per_ep        The maximum timesteps to allow per
                                      episode.
                 batch_size           The batch size to use when training/
                                      updating the networks.
                 ts_per_rollout       A soft limit on the number of timesteps
                                      to allow per rollout (can span multiple
                                      episodes). Note that our actual timestep
                                      count can exceed this limit, but we won't
                                      start any new episodes once it has.
                 gamma                The 'gamma' value for calculating
                                      advantages.
                 lambd                The 'lambda' value for calculating GAEs.
                 epochs_per_iter      'Epoch' is used loosely and with a variety
                                      of meanings in RL. In this case, a single
                                      epoch is a single update of all networks.
                                      epochs_per_iter is the number of updates
                                      to perform after a single rollout (which
                                      may contain multiple episodes).
                 clip                 The clip value to use in the PPO clip.
                 bootstrap_clip       When using GAE, we bootstrap the values
                                      and rewards when an epsiode is cut off
                                      before completion. In these cases, we
                                      clip the bootstrapped reward to a
                                      specific range. Why is this? Well, our
                                      estimated reward (from our value network)
                                      might be way outside of the expected
                                      range.
                 dynamic_bs_clip      If set to True, bootstrap_clip will be
                                      used as the initial clip values, but all
                                      values thereafter will be taken from the
                                      global min and max rewards that have been
                                      seen so far.
                 use_gae              Should we use Generalized Advantage
                                      Estimations? If not, fall back on the
                                      vanilla advantage calculation.
                 use_icm              Should we use an Intrinsic Curiosity
                                      Module?
                 icm_beta             The beta value used within the ICM.
                 ext_reward_weight    An optional weight for the extrinsic
                                      reward.
                 intr_reward_weight   an optional weight for the intrinsic
                                      reward.
                 entropy_weight       An optional weight to apply to our
                                      entropy.
                 target_kl            KL divergence used for early stopping.
                                      This is typically set in the range
                                      [0.1, 0.5]. Use high values to disable.
                                      (Disabled by default).
                 mean_window_size     The window size for a running mean. Note
                                      that each "score" in the window is
                                      actually the mean score for that rollout.
                 render               Should we render the environment while
                                      training?
                 load_state           Should we load a saved state?
                 state_path           The path to save/load our state.
                 save_best_only       When enabled, only save the models when
                                      the top mean window increases. Note that
                                      this assumes that the top scores will be
                                      the best policy, which might not always
                                      hold up, but it's a general assumption.
        """

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
        self.action_squeeze = need_action_squeeze(env)

        if lr_dec == None:
            self.lr_dec = LogDecrementer(
                max_iteration = 2000,
                max_value     = lr,
                min_value     = min_lr)
        else:
            self.lr_dec = lr_dec

        self.env                 = env
        self.device              = device
        self.state_path          = state_path
        self.render              = render
        self.action_type         = action_type
        self.use_gae             = use_gae
        self.use_icm             = use_icm
        self.icm_beta            = icm_beta
        self.ext_reward_weight   = ext_reward_weight
        self.intr_reward_weight  = intr_reward_weight
        self.lr                  = lr
        self.min_lr              = min_lr
        self.max_ts_per_ep       = max_ts_per_ep
        self.batch_size          = batch_size
        self.ts_per_rollout      = ts_per_rollout
        self.gamma               = gamma
        self.lambd               = lambd
        self.target_kl           = target_kl
        self.epochs_per_iter     = epochs_per_iter
        self.clip                = clip
        self.bootstrap_clip      = bootstrap_clip
        self.dynamic_bs_clip     = dynamic_bs_clip
        self.entropy_weight      = entropy_weight
        self.obs_shape           = env.observation_space.shape
        self.prev_top_window     = -np.finfo(np.float32).max
        self.save_best_only      = save_best_only
        self.mean_window_size    = mean_window_size 
        self.score_cache         = np.zeros(0)

        #
        # Create a dictionary to track the status of training.
        #
        max_int           = np.iinfo(np.int32).max
        self.status_dict  = {}
        self.status_dict["iteration"]            = 0
        self.status_dict["longest run"]          = 0
        self.status_dict["window avg"]           = 'N/A'
        self.status_dict["window grad"]          = 'N/A'
        self.status_dict["top window avg"]       = 'N/A'
        self.status_dict["score avg"]            = 0
        self.status_dict["extrinsic score avg"]  = 0
        self.status_dict["top score"]            = -max_int
        self.status_dict["total episodes"]       = 0
        self.status_dict["weighted entropy"]     = 0
        self.status_dict["actor loss"]           = 0
        self.status_dict["critic loss"]          = 0
        self.status_dict["kl avg"]               = 0
        self.status_dict["lr"]                   = self.lr
        self.status_dict["reward range"]         = (max_int, -max_int)

        print("Using {} action type.".format(self.action_type))

        need_softmax = False
        if action_type == "discrete":
            need_softmax = True

        use_conv2d_setup = False
        for base in ac_network.__bases__:
            if base.__name__ == "PPOConv2dNetwork":
                use_conv2d_setup = True

        if use_conv2d_setup:
            obs_dim = self.obs_shape

            self.actor = ac_network(
                "actor", 
                obs_dim, 
                self.act_dim, 
                need_softmax,
                **ac_kw_args)

            self.critic = ac_network(
                "critic", 
                obs_dim, 
                1,
                False,
                **ac_kw_args)

        else:
            obs_dim = self.obs_shape[0]
            self.actor = ac_network(
                "actor", 
                obs_dim, 
                self.act_dim, 
                need_softmax,
                **ac_kw_args)

            self.critic = ac_network(
                "critic", 
                obs_dim, 
                1,
                False,
                **ac_kw_args)

        self.actor  = self.actor.to(device)
        self.critic = self.critic.to(device)

        if self.use_icm:
            self.icm_model = icm_network(
                obs_dim     = obs_dim,
                act_dim     = self.act_dim,
                action_type = self.action_type,
                **icm_kw_args)

            self.icm_model.to(device)
            self.status_dict["icm loss"] = 0
            self.status_dict["intrinsic score avg"] = 0

        if load_state:
            if not os.path.exists(state_path):
                msg  = "WARNING: state_path does not exist. Unable "
                msg += "to load state."
                print(msg)
            else:
                #
                # Let's ensure backwards compatibility with previous commits.
                #
                tmp_status_dict = self.load()

                for key in tmp_status_dict:
                    if key in self.status_dict:
                        self.status_dict[key] = tmp_status_dict[key]

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

            if len(batch_actions.shape) < 2:
                log_probs = dist.log_prob(batch_actions.unsqueeze(1).cpu())
            else:
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
        dataset            = PPODataset(self.device, self.action_type)
        total_episodes     = 0  
        total_ts           = 0
        total_ext_rewards  = 0
        total_intr_rewards = 0
        top_ep_score       = -np.finfo(np.float32).max
        total_score        = 0
        longest_run        = 0
        max_reward         = -np.finfo(np.float32).max
        min_reward         = np.finfo(np.float32).max

        self.actor.eval()
        self.critic.eval()

        while total_ts < self.ts_per_rollout:

            episode_info = EpisodeInfo(
                use_gae        = self.use_gae,
                gamma          = self.gamma,
                lambd          = self.lambd,
                bootstrap_clip = self.bootstrap_clip)

            total_episodes  += 1
            done             = False
            obs              = self.env.reset()
            ep_score         = 0

            #
            # HACK: some environments are buggy and return inconsistent
            # observation shapes. We can enforce shapes here.
            #
            obs = obs.reshape(self.obs_shape)

            for ts in range(self.max_ts_per_ep + 1):
                if self.render:
                    self.env.render()

                total_ts += 1
                action, log_prob = self.get_action(obs)

                t_obs    = torch.tensor(obs, dtype=torch.float).to(self.device)
                t_obs    = t_obs.unsqueeze(0)
                value    = self.critic(t_obs)
                prev_obs = obs.copy()

                if self.action_squeeze:
                    action = action.squeeze()

                obs, ext_reward, done, _ = self.env.step(action)

                if action.size == 1:
                    ep_action = action.item()
                else:
                    ep_action = action

                ep_value = value.item()

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
                ext_reward    *= self.ext_reward_weight

                #
                # If we're using the ICM, we need to do some extra work here.
                # This amounts to adding "curiosity", aka intrinsic reward,
                # to out extrinsic reward.
                #
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
                        intr_reward, _, _ = self.icm_model(obs_1, obs_2, action)

                    intr_reward  = intr_reward.item()
                    intr_reward *= self.intr_reward_weight

                    total_intr_rewards += intr_reward

                    reward = ext_reward + intr_reward

                else:
                    reward = ext_reward

                episode_info.add_info(
                    prev_obs,
                    obs,
                    ep_action,
                    ep_value,
                    log_prob,
                    reward)

                max_reward         = max(max_reward, reward)
                min_reward         = min(min_reward, reward)
                total_score       += reward
                ep_score          += natural_reward
                total_ext_rewards += natural_reward

                if done:
                    #
                    # Avoid clipping the reward here in case our clip range
                    # doesn't include 0.
                    #
                    episode_info.end_episode(
                        ending_value   = 0,
                        episode_length = ts + 1,
                        skip_clip      = True)
                    break

                elif ts == self.max_ts_per_ep:
                    t_obs     = torch.tensor(obs, dtype=torch.float).to(self.device)
                    t_obs     = t_obs.unsqueeze(0)
                    nxt_value = self.critic(t_obs)

                    episode_info.end_episode(
                        ending_value   = nxt_value.item(),
                        episode_length = ts + 1,
                        skip_clip      = False)

            dataset.add_episode(episode_info)
            top_ep_score = max(top_ep_score, ep_score)
            longest_run  = max(longest_run, ts + 1)

        #
        # Update our status dict.
        #
        top_score          = max(top_ep_score, self.status_dict["top score"])
        running_ext_score  = total_ext_rewards / total_episodes
        running_score      = total_score / total_episodes

        self.status_dict["score avg"]            = running_score
        self.status_dict["extrinsic score avg"]  = running_ext_score
        self.status_dict["top score"]            = top_score
        self.status_dict["total episodes"]       = total_episodes
        self.status_dict["longest run"]          = longest_run

        max_reward = max(self.status_dict["reward range"][1], max_reward)
        min_reward = min(self.status_dict["reward range"][0], min_reward)
        rw_range   = (min_reward, max_reward)
        self.status_dict["reward range"] = rw_range

        if self.dynamic_bs_clip:
            self.bootstrap_clip = rw_range

        #
        # Update our score cache and the window mean when appropriate.
        #
        if self.score_cache.size < self.mean_window_size:
            self.score_cache = np.append(self.score_cache, running_score)
            self.status_dict["window avg"]  = "N/A"
            self.status_dict["window grad"] = "N/A"
        else:
            self.score_cache = np.roll(self.score_cache, -1)
            self.score_cache[-1] = running_score

            self.status_dict["window avg"]  = self.score_cache.mean()

            w_grad = np.gradient(self.score_cache).mean()
            self.status_dict["window grad"] = w_grad

            if type(self.status_dict["top window avg"]) == str:
                top_window = self.status_dict["window avg"]
            else:
                top_window = max(self.status_dict["window avg"],
                    self.status_dict["top window avg"])

            self.status_dict["top window avg"] = top_window
            self.prev_top_window = top_window

        if self.use_icm:
            ism = total_intr_rewards / total_episodes
            self.status_dict["intrinsic score avg"] = ism

        #
        # Finally, bulid the dataset.
        #
        dataset.build()

        return dataset

    def learn(self, total_timesteps):
        start_time = time.time()
        ts = 0

        while ts < total_timesteps:
            dataset = self.rollout()

            ts += np.sum(dataset.ep_lens)
            self.status_dict["iteration"] += 1

            self.lr = self.lr_dec.decrement(self.status_dict["iteration"])
            self.status_dict["lr"] = self.lr

            data_loader = DataLoader(
                dataset,
                batch_size = self.batch_size,
                shuffle    = True)

            self.actor.train()
            self.critic.train()

            for i in range(self.epochs_per_iter):

                self._ppo_batch_train(data_loader)

                if self.status_dict["kl avg"] > (1.5 * self.target_kl):
                    msg  = "\nTarget KL of {} ".format(1.5 * self.target_kl)
                    msg += "has been reached. "
                    msg += "Ending early (after {} epochs)".format(i + 1)
                    print(msg)
                    break

            if self.use_icm:
                for _ in range(self.epochs_per_iter):
                    self._icm_batch_train(data_loader)

            self.print_status()

            if type(self.status_dict["top window avg"]) == str:
                self.save()
            elif (self.save_best_only and
                self.prev_top_window <= self.status_dict["top window avg"]):
                self.save()
            elif not self.save_best_only:
                self.save()

        stop_time = time.time()
        minutes   = (stop_time - start_time) / 60.
        print("Time spent training: {} minutes".format(minutes))

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
            # TODO: what's with the exponent?
            #
            ratios    = torch.exp(curr_log_probs - log_probs)
            surr1     = ratios * advantages
            surr2     = torch.clamp(
                ratios, 1 - self.clip, 1 + self.clip) * advantages
            total_kl += (log_probs - curr_log_probs).mean().item()

            #
            # We negate here to perform gradient ascent rather than descent.
            #
            actor_loss        = (-torch.min(surr1, surr2)).mean()
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

        self.status_dict["weighted entropy"] = w_entropy / counter
        self.status_dict["actor loss"]       = total_actor_loss / counter
        self.status_dict["critic loss"]      = total_critic_loss / counter
        self.status_dict["kl avg"]           = total_kl / counter

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
            tmp_status_dict = pickle.load(in_f)

        return tmp_status_dict

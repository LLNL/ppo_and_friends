from .utils import RunningMeanStd
import numpy as np
import pickle
import os

class IdentityWrapper(object):

    def __init__(self,
                 env,
                 **kw_args):

        self.env               = env
        self.observation_space = env.observation_space
        self.action_space      = env.action_space

    def step(self, action):
        return self.env.step(action)

    def reset(self):
        return self.env.reset()

    def render(self):
        self.env.render()

    def save_info(self, path):
        self._check_env_save(path)

    def load_info(self, path):
        self._check_env_load(path)

    def _check_env_save(self, path):
        save_info = getattr(self.env, "save_info", None)

        if callable(save_info):
            save_info(path)

    def _check_env_load(self, path):
        load_info = getattr(self.env, "load_info", None)

        if callable(load_info):
            load_info(path)


class ObservationNormalizer(IdentityWrapper):

    def __init__(self,
                 env,
                 update_stats = True,
                 epsilon      = 1e-8,
                 **kw_args):

        super(ObservationNormalizer, self).__init__(
            env,
            **kw_args)

        self.running_stats = RunningMeanStd(
            shape = self.env.observation_space.shape)

        self.update_stats  = update_stats
        self.epsilon       = epsilon

    def step(self, action):
        obs, reward, done, info = self.env.step(action)

        if self.update_stats:
            self.running_stats.update(obs)

        obs = self.normalize(obs)

        return obs, reward, done, info

    def reset(self):
        obs = self.env.reset()

        if self.update_stats:
            self.running_stats.update(obs)

        return obs

    def normalize(self, obs):
        obs = (obs - self.running_stats.mean) / \
            np.sqrt(self.running_stats.variance + self.epsilon)
        return obs

    def save_info(self, path):
        out_file = os.path.join(path, "RunningObsStats.pkl")

        with open(out_file, "wb") as fh:
            pickle.dump(self.running_stats, fh)

        self._check_env_save(path)

    def load_info(self, path):
        in_file = os.path.join(path, "RunningObsStats.pkl")

        with open(in_file, "rb") as fh:
            self.running_stats = pickle.load(fh)

        self._check_env_load(path)


class RewardNormalizer(IdentityWrapper):

    def __init__(self,
                 env,
                 update_stats = True,
                 epsilon      = 1e-8,
                 gamma        = 0.99,
                 **kw_args):

        super(RewardNormalizer, self).__init__(
            env,
            **kw_args)

        self.running_stats = RunningMeanStd(shape=())

        self.update_stats   = update_stats
        self.epsilon        = epsilon
        self.gamma          = gamma
        self.running_reward = np.zeros(1)

    def step(self, action):

        obs, reward, done, info = self.env.step(action)

        if self.update_stats:

            self.running_reward[0] = self.running_reward * self.gamma + reward
            self.running_stats.update(self.running_reward)

        if done:
            self.running_reward[0] = 0.0

        if "natural reward" not in info:
            info["natural reward"] = reward

        reward = self.normalize(reward)

        return obs, reward, done, info

    def normalize(self, reward):
        reward /= np.sqrt(self.running_stats.variance + self.epsilon)
        return reward

    def save_info(self, path):
        out_file = os.path.join(path, "RunningRewardStats.pkl")

        with open(out_file, "wb") as fh:
            pickle.dump(self.running_stats, fh)

        self._check_env_save(path)

    def load_info(self, path):
        in_file = os.path.join(path, "RunningRewardStats.pkl")

        with open(in_file, "rb") as fh:
            self.running_stats = pickle.load(fh)

        self._check_env_load(path)


class ObservationClipper(IdentityWrapper):

    def __init__(self,
                 env,
                 clip_range = (-10., 10.),
                 **kw_args):

        super(ObservationClipper, self).__init__(
            env,
            **kw_args)

        self.clip_range = clip_range

    def step(self, action):
        obs, reward, done, info = self.env.step(action)

        obs = self._clip(obs)

        return obs, reward, done, info

    def reset(self):
        return self._clip(self.env.reset())

    def _clip(self, obs):
        return np.clip(obs, self.clip_range[0], self.clip_range[1])


class RewardClipper(IdentityWrapper):

    def __init__(self,
                 env,
                 clip_range = (-10., 10.),
                 **kw_args):

        super(RewardClipper, self).__init__(
            env,
            **kw_args)

        self.clip_range = clip_range

    def step(self, action):
        obs, reward, done, info = self.env.step(action)

        if "natural reward" not in info:
            info["natural reward"] = reward

        reward = self._clip(reward)

        return obs, reward, done, info

    def _clip(self, reward):
        return np.clip(reward, self.clip_range[0], self.clip_range[1])


"""
    A home for generic environment wrappers. The should not be
    specific to any type of environment.
"""
from ppo_and_friends.utils.stats import RunningMeanStd
import numpy as np
import pickle
import os
from mpi4py import MPI

comm      = MPI.COMM_WORLD
rank      = comm.Get_rank()
num_procs = comm.Get_size()


class IdentityWrapper(object):
    """
        A wrapper that acts exactly like the original environment but also
        has a few extra bells and whistles.
    """

    def __init__(self,
                 env,
                 **kw_args):
        """
            Initialize the wrapper.

            Arguments:
                env    The environment to wrap.
        """

        self.env               = env
        self.observation_space = env.observation_space
        self.action_space      = env.action_space
        self.need_hard_reset   = True
        self.obs_cache         = None

    def step(self, action):
        """
            Take a single step in the environment using the given
            action.

            Arguments:
                action    The action to take.

            Returns:
                The resulting observation, reward, done, and info tuple.
        """
        obs, reward, done, info = self.env.step(action)

        #
        # HACK: some environments are buggy and don't follow their
        # own rules!
        #
        obs = obs.reshape(self.observation_space.shape)

        self.obs_cache = obs.copy()
        self.need_hard_reset = False

        return obs, reward, done, info

    def reset(self):
        """
            Reset the environment.

            Returns:
                The resulting observation.
        """
        obs = self.env.reset()
        obs = obs.reshape(self.observation_space.shape)
        return obs

    def soft_reset(self):
        """
            Perform a "soft reset". This results in only performing the reset
            if the environment hasn't been reset since being created. This can
            allow us to start a new rollout from a previous rollout that ended
            near later in the environments timesteps.

            Returns:
                An observation.
        """
        if self.need_hard_reset:
            return self.reset()
        return self.obs_cache

    def render(self, **kw_args):
        """
            Render the environment.
        """
        return self.env.render(**kw_args)

    def save_info(self, path):
        """
            Save any info needed for loading the environment at a later time.

            Arguments:
                path    The path to save to.
        """
        self._check_env_save(path)

    def load_info(self, path):
        """
            Load any info needed for reinstating a saved environment.

            Arguments:
                path    The path to load from.
        """
        self._check_env_load(path)

    def _check_env_save(self, path):
        """
            Determine if our wrapped environment has a "save_info"
            method. If so, call it.

            Arguments:
                path    The path to save to.
        """
        save_info = getattr(self.env, "save_info", None)

        if callable(save_info):
            save_info(path)

    def _check_env_load(self, path):
        """
            Determine if our wrapped environment has a "load_info"
            method. If so, call it.

            Arguments:
                path    The path to load from.
        """
        load_info = getattr(self.env, "load_info", None)

        if callable(load_info):
            load_info(path)


class ObservationNormalizer(IdentityWrapper):
    """
        A wrapper for normalizing observations. This normalization
        method uses running statistics.
        NOTE: this uses implentations very similar to some of Stable
        Baslines' VecNormalize.
    """

    def __init__(self,
                 env,
                 update_stats = True,
                 epsilon      = 1e-8,
                 **kw_args):
        """
            Initialize the wrapper.

            Arguments:
                env            The environment to wrap.
                update_stats   Should we update the statistics? We typically
                               have this enabled until testing.
                epsilon        A very small number to help avoid dividing by 0.
        """

        super(ObservationNormalizer, self).__init__(
            env,
            **kw_args)

        self.running_stats = RunningMeanStd(
            shape = self.env.observation_space.shape)

        self.update_stats  = update_stats
        self.epsilon       = epsilon

    def step(self, action):
        """
            Take a single step in the environment using the given
            action, update the running stats, and normalize the
            resulting observation.

            Arguments:
                action    The action to take.

            Returns:
                The resulting observation, reward, done, and info tuple.
        """
        obs, reward, done, info = self.env.step(action)

        if self.update_stats:
            self.running_stats.update(obs)

        obs = self.normalize(obs)

        return obs, reward, done, info

    def reset(self):
        """
            Reset the environment, and update the running stats.

            Returns:
                The resulting observation.
        """
        obs = self.env.reset()

        if self.update_stats:
            self.running_stats.update(obs)

        return obs

    def normalize(self, obs):
        """
            Normalize an observation using our running stats.

            Arguments:
                obs    The observation to normalize.

            Returns:
                The normalized observation.
        """
        obs = (obs - self.running_stats.mean) / \
            np.sqrt(self.running_stats.variance + self.epsilon)
        return obs

    def save_info(self, path):
        """
            Save out our running stats, and check if our wrapped
            environment needs to perform any more info saves.

            Arguments:
                path    The path to save to.
        """
        file_name = "RunningObsStats_{}.pkl".format(rank)
        out_file  = os.path.join(path, file_name)

        with open(out_file, "wb") as fh:
            pickle.dump(self.running_stats, fh)

        self._check_env_save(path)

    def load_info(self, path):
        """
            Load our running stats and check to see if our wrapped
            environment needs to load anything.

            Arguments:
                path    The path to load from.
        """
        file_name = "RunningObsStats_{}.pkl".format(rank)
        in_file   = os.path.join(path, file_name)

        with open(in_file, "rb") as fh:
            self.running_stats = pickle.load(fh)

        self._check_env_load(path)


class RewardNormalizer(IdentityWrapper):
    """
        This wrapper uses running statistics to normalize rewards.
        NOTE: much of this logic comes from Stable Baseline's
        VecNormalize.
    """

    def __init__(self,
                 env,
                 update_stats = True,
                 epsilon      = 1e-8,
                 gamma        = 0.99,
                 **kw_args):

        super(RewardNormalizer, self).__init__(
            env,
            **kw_args)
        """
            Initialize the wrapper.

            Arguments:
                env            The environment to wrap.
                update_stats   Whether or not to update our running stats. This
                               is typically set to true until testing.
                epsilon        A very small value to help avoid 0 divisions.
                gamma          A discount factor for our running stats.
        """

        self.running_stats = RunningMeanStd(shape=())

        self.update_stats   = update_stats
        self.epsilon        = epsilon
        self.gamma          = gamma
        self.running_reward = np.zeros(1)

    def step(self, action):
        """
            Take a single step in the environment using the given
            action, update the running stats, and normalize the
            resulting reward.

            Arguments:
                action    The action to take.

            Returns:
                The resulting observation, reward, done, and info tuple.
        """

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
        """
            Normalize our reward using Stable Baseline's approach.

            Arguments:
                reward    The reward to normalize.

            Returns:
                The normalized reward.
        """
        reward /= np.sqrt(self.running_stats.variance + self.epsilon)
        return reward

    def save_info(self, path):
        """
            Save out our running stats, and check if our wrapped
            environment needs to perform any more info saves.

            Arguments:
                path    The path to save to.
        """
        file_name = "RunningRewardsStats_{}.pkl".format(rank)
        out_file  = os.path.join(path, file_name)

        with open(out_file, "wb") as fh:
            pickle.dump(self.running_stats, fh)

        self._check_env_save(path)

    def load_info(self, path):
        """
            Load our running stats and check to see if our wrapped
            environment needs to load anything.

            Arguments:
                path    The path to load from.
        """
        file_name = "RunningRewardsStats_{}.pkl".format(rank)
        in_file   = os.path.join(path, file_name)

        with open(in_file, "rb") as fh:
            self.running_stats = pickle.load(fh)

        self._check_env_load(path)


class ObservationClipper(IdentityWrapper):
    """
        An environment wrapper that clips observations.
    """

    def __init__(self,
                 env,
                 clip_range = (-10., 10.),
                 **kw_args):
        """
            Initialize the wrapper.

            Arguments:
                env         The environment to wrap.
                clip_range  The range to clip our observations into.
        """

        super(ObservationClipper, self).__init__(
            env,
            **kw_args)

        self.clip_range = clip_range

    def step(self, action):
        """
            Take a single step in the environment using the given
            action, and clip the resulting observation.

            Arguments:
                action    The action to take.

            Returns:
                The resulting observation, reward, done, and info tuple.
        """
        obs, reward, done, info = self.env.step(action)

        obs = self._clip(obs)

        return obs, reward, done, info

    def reset(self):
        """
            Reset the environment, and clip the observation.

            Returns:
                The resulting observation.
        """
        return self._clip(self.env.reset())

    def _clip(self, obs):
        """
            Perform the observation clip.

            Arguments:
                obs    The observation to clip.

            Returns:
                The clipped observation.
        """
        return np.clip(obs, self.clip_range[0], self.clip_range[1])


class RewardClipper(IdentityWrapper):
    """
        A wrapper for clipping rewards.
    """

    def __init__(self,
                 env,
                 clip_range = (-10., 10.),
                 **kw_args):
        """
            Initialize the wrapper.

            Arguments:
                env         The environment to wrap.
                clip_range  The range to clip our rewards into.
        """

        super(RewardClipper, self).__init__(
            env,
            **kw_args)

        self.clip_range = clip_range

    def step(self, action):
        """
            Take a single step in the environment using the given
            action, and clip the resulting reward.

            Arguments:
                action    The action to take.

            Returns:
                The resulting observation, reward, done, and info tuple.
        """
        obs, reward, done, info = self.env.step(action)

        if "natural reward" not in info:
            info["natural reward"] = reward

        reward = self._clip(reward)

        return obs, reward, done, info

    def _clip(self, reward):
        """
            Perform the reward clip.

            Arguments:
                reward    The reward to clip.

            Returns:
                The clipped reward.
        """
        return np.clip(reward, self.clip_range[0], self.clip_range[1])


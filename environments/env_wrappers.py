"""
    A home for generic environment wrappers. The should not be
    specific to any type of environment.
"""
from ppo_and_friends.utils.stats import RunningMeanStd
import numpy as np
import pickle
import os
from ppo_and_friends.utils.mpi_utils import rank_print
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
                env           The environment to wrap.
                status_dict   The dictionary containing training stats.
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


class VectorizedEnv(IdentityWrapper):
    """
        Wrapper for "vectorizing" environments. As far as I can tell, this
        definition of vectorize is pretty specific to RL and refers to a way
        of reducing our environment episodes to "fixed-length trajecorty
        segments". This allows us to set our maximum timesteps per episode
        fairly low while still observing an episode from start to finish.
        This is accomplished by these "segments" that can (and often do) start
        in the middle of an episode.
        Vectorized environments are also often referred to as environment
        wrappers that contain multiple instances of environments, which then
        return batches of observations, rewards, etc. We don't currently
        support this second idea.
    """

    def __init__(self,
                 env,
                 **kw_args):
        """
            Initialize our vectorized environment.

            Arguments:
                env    The environment to vectorize.
        """
        super(VectorizedEnv, self).__init__(
            env,
            **kw_args)

    def step(self, action):
        """
            Take a step in our environment with the given action.
            Since we're vectorized, we reset the environment when
            we've reached a "done" state.

            Arguments:
                action    The action to take.

            Returns:
                The resulting observation, reward, done, and info
                tuple.
        """
        obs, reward, done, info = self.env.step(action)

        #
        # HACK: some environments are buggy and don't follow their
        # own rules!
        #
        obs = obs.reshape(self.observation_space.shape)

        if type(reward) == np.ndarray:
            reward = reward[0]

        if done:
            info["terminal observation"] = obs.copy()
            obs = self.env.reset()
            obs = obs.reshape(self.observation_space.shape)

        return obs, reward, done, info


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


class GenericClipper(IdentityWrapper):
    """
        A wrapper for clipping rewards.
    """

    def __init__(self,
                 env,
                 status_dict = {},
                 clip_range  = (-10., 10.),
                 **kw_args):
        """
            Initialize the wrapper.

            Arguments:
                env         The environment to wrap.
                status_dict The training status dictionary. This is used when
                            our clip range contains callables.
                clip_range  The range to clip our rewards into. This can be
                            either a real number or a class that inherits from
                            IterationMapper.
        """

        super(GenericClipper, self).__init__(
            env,
            **kw_args)

        min_callable = None
        max_callable = None

        self.need_iteration = False
        self.status_dict    = status_dict

        if callable(clip_range[0]):
            min_callable = clip_range[0]
            self.need_iteration = True
        else:
            min_callable = lambda x : clip_range[0]

        if callable(clip_range[1]):
            max_callable = clip_range[1]
            self.need_iteration = True
        else:
            max_callable = lambda x : clip_range[1]

        self.clip_range = (min_callable, max_callable)

    def get_clip_range(self):
        """
            Get the current clip range.

            Returns:
                A tuple containing the clip range as (min, max).
        """
        if self.need_iteration:
            if "iteration" not in self.status_dict:
                msg  = "ERROR: clipper requires 'iteration' from the status "
                msg += "dictionary, but it's not there. "
                msg += "\nstatus_dict: \n{}".format(self.status_dict)
                rank_print(msg)
                comm.Abort()

            min_value = self.clip_range[0](self.status_dict["iteration"])
            max_value = self.clip_range[1](self.status_dict["iteration"])
        else:
            min_value = self.clip_range[0](None)
            max_value = self.clip_range[1](None)

        return (min_value, max_value)

    def _clip(self, val):
        """
            Perform the clip.

            Arguments:
                val    The value to be clipped.
        """
        raise NotImplementedError


class ObservationClipper(GenericClipper):
    """
        An environment wrapper that clips observations.
    """

    def __init__(self,
                 env,
                 status_dict = {},
                 clip_range  = (-10., 10.),
                 **kw_args):
        """
            Initialize the wrapper.

            Arguments:
                env         The environment to wrap.
                status_dict The training status dictionary. This is used when
                            our clip range contains callables.
                clip_range  The range to clip our observations into.
        """
        super(ObservationClipper, self).__init__(
            env,
            status_dict = status_dict,
            clip_range  = clip_range,
            **kw_args)

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
        min_value, max_value = self.get_clip_range()
        return np.clip(obs, min_value, max_value)


class RewardClipper(GenericClipper):
    """
        A wrapper for clipping rewards.
    """

    def __init__(self,
                 env,
                 status_dict = {},
                 clip_range  = (-10., 10.),
                 **kw_args):
        """
            Initialize the wrapper.

            Arguments:
                env         The environment to wrap.
                status_dict The training status dictionary. This is used when
                            our clip range contains callables.
                clip_range  The range to clip our rewards into.
        """

        super(RewardClipper, self).__init__(
            env,
            status_dict = status_dict,
            clip_range  = clip_range,
            **kw_args)

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
        min_value, max_value = self.get_clip_range()
        return np.clip(reward, min_value, max_value)


class AugmentingEnvWrapper(IdentityWrapper):
    """
    """

    def __init__(self,
                 env,
                 **kw_args):
        """
            Initialize the wrapper.

            Arguments:
                env            The environment to wrap.
        """

        super(ObservationNormalizer, self).__init__(
            env,
            **kw_args)

        aug_func = getattr(env, "augment_observation", None)

        if type(aug_func) == type(None):
            msg  = "ERROR: env must define 'augment_observation' in order "
            msg += "to qualify for the AugmentingEnvWrapper class."
            rank_print(msg)
            comm.Abort()

    def step(self, action):
        """
            Take a single step in the environment using the given
            action, allow the environment to augment the returned
            observation, and set up the return values as a batch.

            Arguments:
                action    The action to take.

            Returns:
                A batch of observations, rewards, and dones along
                with a single info dictionary. The observations will
                contain augmentations of the original observation.
        """
        obs, reward, done, info = self.env.step(action)

        aug_obs_batch = self.env.augment_observation(obs)
        batch_size    = aug_obs_batch.shape[0]

        if "terminal observation" in info:
            terminal_obs = info["terminal observation"]
            terminal_obs = self.env.augment_observation(terminal_obs)
            info["terminal observation"] = terminal_obs

        rewards = np.tile(np.array((reward,)), batch_size).reshape((batch_size, 1))
        dones   = np.tile(np.array((done,)), batch_size).reshape((batch_size, 1))

        return aug_obs_batch, rewards, dones, info

    def reset(self):
        """
            Reset the environment, and update the running stats.

            Returns:
                The resulting observation.
        """
        obs = self.env.reset()

        aug_obs_batch = self.env.augment_observation(obs)

        return aug_obs_batch

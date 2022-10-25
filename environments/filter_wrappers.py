"""
    Environment wrappers that send data through filtration of
    some kind. This can be things like normalization, clipping,
    augmenting, etc.
"""
from ppo_and_friends.utils.stats import RunningMeanStd
from ppo_and_friends.environments.general_wrappers import IdentityWrapper
import numpy as np
import pickle
import os
from ppo_and_friends.utils.mpi_utils import rank_print
from abc import ABC, abstractmethod
import sys

from mpi4py import MPI
comm      = MPI.COMM_WORLD
rank      = comm.Get_rank()
num_procs = comm.Get_size()

class ObservationFilter(IdentityWrapper, ABC):
    """
        An abstract class for filtering observations.
    """

    @abstractmethod
    def _filter_global_observation(self, global_obs):
        """
            Perform the local filtering on the global observation.

            Arguments:
                global_obs    The global observation to filter.

            Returns:
                The filtered observation.
        """
        return

    @abstractmethod
    def _filter_local_observation(self, obs):
        """
            Perform the local filtering on the local observation.

            Arguments:
                obs    The local observation to filter.

            Returns:
                The filtered observation.
        """
        return

    def step(self, action):
        """
            Take a single step in the environment using the given
            action, update the running stats, and filter the
            resulting observation.

            Arguments:
                action    The action to take.

            Returns:
                The resulting observation, reward, done, and info tuple.
        """
        obs, reward, done, info = self.env.step(action)

        obs = self._filter_local_observation(obs)

        #
        # We need to cache the observation in case our lower level
        # wrappers don't have soft_reset defined.
        #
        self.obs_cache = obs.copy()
        self.need_hard_reset = False

        #
        # Info can come back int two forms:
        #   1. It's a dictionary containing global information.
        #   2. It's an iterable containing num_agent dictionaries,
        #      each of which contains info for its assocated agent.
        #
        # In either case, we need to check for global state. If it's there
        # we need to apply the same filters that we're applying to the
        # local observations.
        #
        info_is_global = False
        if type(info) == dict:
            info_is_global = True

        if info_is_global and "global state" in info:
            info["global state"] = \
                self._filter_global_observation(info["global state"])
        else:
            #
            # If it's in one, it's in them all.
            #
            if "global state" in info[0]:

                iter_size = len(info)
                for i in range(iter_size):
                    info[i]["global state"] = \
                        self._filter_global_observation(info[i]["global state"])

        return obs, reward, done, info

    def reset(self):
        """
            Reset the environment, and perform any needed filtering.

            Returns:
                The resulting observation.
        """
        obs = self.env.reset()

        #
        # In our multi-agent case, the global state is returned along
        # with the local observation.
        #
        is_multi_agent = False
        if type(obs) == tuple  or type(obs) == list:
            is_multi_agent = True
            obs, global_state = obs

        obs = self._filter_local_observation(obs)

        if is_multi_agent:
            global_state = self._filter_global_observation(global_state)
            return (obs, global_state)

        return obs

    def soft_reset(self):
        """
            Reset the environment, and perform any needed filtering.

            Returns:
                The resulting observation.
        """
        obs = self._env_soft_reset()

        #
        # In our multi-agent case, the global state is returned along
        # with the local observation.
        #
        is_multi_agent = False
        if type(obs) == tuple  or type(obs) == list:
            is_multi_agent = True
            obs, global_state = obs

        obs = self._filter_local_observation(obs)

        if is_multi_agent:
            global_state = self._filter_global_observation(global_state)
            return (obs, global_state)

        return obs


class ObservationNormalizer(ObservationFilter):
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

        #
        # We might need to normalize global observations separately, so
        # let's keep track of a separate running stats for this.
        #
        global_obs_shape = self.env.get_global_state_space().shape

        self.global_running_stats = RunningMeanStd(
            shape = global_obs_shape)

        self.update_stats  = update_stats
        self.epsilon       = epsilon

    def _filter_local_observation(self, obs):
        """
            Send a local observation through the normalization process.

            Arguments:
                obs    The observation to normalize.

            Returns:
                The normalized observation.
        """
        if self.update_stats:
            self.running_stats.update(obs)

        return self.local_normalize(obs)

    def _filter_global_observation(self, global_obs):
        """
            Send a global observation through the normalization process. This may
            called recursively by filter_global_observation().

            Arguments:
                global_obs    The global observation to normalize.

            Returns:
                The normalized observation.
        """
        if self.update_stats:
            self.global_running_stats.update(global_obs)

        return self.global_normalize(global_obs)

    def local_normalize(self, obs):
        """
            A simple wrapper around self._local_normalize() that mitigates
            issues with memory references.

            Arguments:
                obs    The observation to normalize.

            Returns:
                The normalized observation.
        """
        if type(obs) == np.ndarray:
            return self._local_normalize(obs.copy())
        return self._local_normalize(obs)

    def _local_normalize(self, obs):
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

    def global_normalize(self, global_obs):
        """
            A simple wrapper around self._global_normalize() that mitigates
            issues with memory references.

            Arguments:
                global_obs    The global observation to normalize.

            Returns:
                The normalized observation.
        """
        if type(global_obs) == np.ndarray:
            return self._global_normalize(global_obs.copy())
        return self._global_normalize(global_obs)

    def _global_normalize(self, global_obs):
        """
            Normalize a global observation using our running stats.

            Arguments:
                global_obs    The global observation to normalize.

            Returns:
                The normalized observation.
        """
        global_obs = (global_obs - self.global_running_stats.mean) / \
            np.sqrt(self.global_running_stats.variance + self.epsilon)
        return global_obs

    def save_info(self, path):
        """
            Save out our running stats, and check if our wrapped
            environment needs to perform any more info saves.

            Arguments:
                path    The path to save to.
        """
        if self.test_mode:
            return

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
        if self.test_mode:
            file_name = "RunningObsStats_0.pkl"
        else:
            file_name = "RunningObsStats_{}.pkl".format(rank)

        in_file = os.path.join(path, file_name)

        #
        # There are cases where we initially train using X ranks, and we
        # later want to continue training using (X+k) ranks. In these cases,
        # let's copy rank 0's info to all ranks > X.
        #
        if not os.path.exists(in_file):
            file_name = "RunningObsStats_0.pkl"
            in_file   = os.path.join(path, file_name)

        with open(in_file, "rb") as fh:
            self.running_stats = pickle.load(fh)

        self._check_env_load(path)


class RewardNormalizer(IdentityWrapper):
    """
        This wrapper uses running statistics to normalize rewards.
        NOTE: some of this logic comes from Stable Baseline's
        VecNormalize.
    """

    def __init__(self,
                 env,
                 update_stats = True,
                 epsilon      = 1e-8,
                 gamma        = 0.99,
                 **kw_args):
        """
            Initialize the wrapper.

            Arguments:
                env            The environment to wrap.
                update_stats   Whether or not to update our running stats. This
                               is typically set to true until testing.
                epsilon        A very small value to help avoid 0 divisions.
                gamma          A discount factor for our running stats.
        """
        super(RewardNormalizer, self).__init__(
            env,
            **kw_args)

        self.running_stats = RunningMeanStd(shape=())
        self.update_stats  = update_stats
        self.epsilon       = epsilon
        self.gamma         = gamma

        #
        # We might be wrapping an environment that supports batches.
        # if so, we need to be able to correlate a running reward with
        # each environment instance.
        #
        if self.test_mode and self.get_num_agents() == 1:
            self.batch_size = 1
        else:
            self.batch_size = self.get_batch_size()

        self.running_reward = np.zeros(self.batch_size)

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

        for env_idx in range(self.batch_size):
            if self.update_stats:
                self.running_reward[env_idx] = \
                    self.running_reward[env_idx] * self.gamma + reward[env_idx]

                self.running_stats.update(self.running_reward)

        if self.batch_size > 1:
            where_done = np.where(done)[0]
            self.running_reward[where_done] = 0.0
        elif done:
            self.running_reward[0] = 0.0

        if type(reward) == np.ndarray:
            batch_size = reward.shape[0]

            for r_idx in range(batch_size):
                if "natural reward" not in info[r_idx]:
                    info[r_idx]["natural reward"] = reward[r_idx]
        else:
            if "natural reward" not in info:
                info["natural reward"] = reward

        reward = self.normalize(reward)

        return obs, reward, done, info

    def normalize(self, reward):
        """
            A simple wrapper around self._normalize() that mitigates
            issues with memory references.

            Arguments:
                reward    The reward to normalize.

            Returns:
                The normalized reward.
        """
        if type(reward) == np.ndarray:
            return self._normalize(reward.copy())
        return self._normalize(reward)

    def _normalize(self, reward):
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
        if self.test_mode:
            return

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
        if self.test_mode:
            file_name = "RunningRewardsStats_0.pkl"
        else:
            file_name = "RunningRewardsStats_{}.pkl".format(rank)

        in_file = os.path.join(path, file_name)

        #
        # There are cases where we initially train using X ranks, and we
        # later want to continue training using (X+k) ranks. In these cases,
        # let's copy rank 0's info to all ranks > X.
        #
        if not os.path.exists(in_file):
            file_name = "RunningRewardsStats_0.pkl"
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
        self.status_dict = status_dict

        if callable(clip_range[0]):
            min_callable = clip_range[0]
        else:
            min_callable = lambda *args, **kwargs : clip_range[0]

        if callable(clip_range[1]):
            max_callable = clip_range[1]
        else:
            max_callable = lambda *args, **kwargs : clip_range[1]

        self.clip_range = (min_callable, max_callable)

    def get_clip_range(self):
        """
            Get the current clip range.

            Returns:
                A tuple containing the clip range as (min, max).
        """
        min_value = self.clip_range[0](
            iteration = self.status_dict["iteration"],
            timestep  = self.status_dict["timesteps"])
        max_value = self.clip_range[1](
            iteration = self.status_dict["iteration"],
            timestep  = self.status_dict["timesteps"])

        return (min_value, max_value)

    def _clip(self, val):
        """
            Perform the clip.

            Arguments:
                val    The value to be clipped.
        """
        raise NotImplementedError


class ObservationClipper(ObservationFilter, GenericClipper):
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

    def _filter_global_observation(self, global_obs):
        """
            A simple wrapper for clipping the global observation.

            Arguments:
                obs    The observation to clip.

            Returns:
                The clipped observation.
        """
        return self._clip(global_obs)

    def _filter_local_observation(self, obs):
        """
            A simple wrapper for clipping local the observation.

            Arguments:
                obs    The observation to clip.

            Returns:
                The clipped observation.
        """
        return self._clip(obs)

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

        if type(reward) == np.ndarray:
            batch_size = reward.shape[0]

            for r_idx in range(batch_size):
                if "natural reward" not in info[r_idx]:
                    info[r_idx]["natural reward"] = reward[r_idx]
        else:
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


class ObservationAugmentingWrapper(IdentityWrapper):
    """
        This wrapper expects the environment to have a method named
        'augment_observation' that can be utilized to augment an observation
        to create a batch of obserations. Each instance of observation in the
        batch will be coupled with an identical done and reward. This is to
        help a policy learn that a particular augmentation does not affect
        the learned policy.
    """

    def __init__(self,
                 env,
                 **kw_args):
        """
            Initialize the wrapper.

            Arguments:
                env   The environment to wrap.
        """

        super(ObservationAugmentingWrapper, self).__init__(
            env,
            **kw_args)

        self.test_idx = -1
        aug_func      = getattr(env, "augment_observation", None)

        if type(aug_func) == type(None):
            msg  = "ERROR: env must define 'augment_observation' in order "
            msg += "to qualify for the ObservationAugmentingWrapper class."
            rank_print(msg)
            comm.Abort()

        self.batch_size = self.aug_reset().shape[0]

    def step(self, action):
        """
            Take a single step in the environment using the given
            action. If we're in test mode, we don't augment. Otherwise,
            call the aug_step method.

            NOTE: the action is expected to be a SINGLE action. This does
            not currently support multiple environment instances.

            Arguments:
                action    The action to take.

            Returns:
                The resulting observation(s), reward(s), done(s), and info(s).
        """
        if self.test_mode:
            return self.aug_test_step(action)
        return self.aug_step(action)

    def aug_step(self, action):
        """
            Take a single step in the environment using the given
            action, allow the environment to augment the returned
            observation, and set up the return values as a batch.

            NOTE: the action is expected to be a SINGLE action. This does
            not currently support multiple environment instances.

            Arguments:
                action    The action to take.

            Returns:
                A batch of observations, rewards, and dones along
                with a single info dictionary. The observations will
                contain augmentations of the original observation.
        """
        obs, reward, done, info = self.env.step(action)

        batch_obs  = self.env.augment_observation(obs)
        batch_size = batch_obs.shape[0]

        if "terminal observation" in info[0]:
            batch_infos = np.array([None] * batch_size,
                dtype=object)

            terminal_obs = info[0]["terminal observation"]
            terminal_obs = self.env.augment_observation(terminal_obs)

            for i in range(batch_size):
                i_info = info[0].copy()
                i_info["terminal observation"] = terminal_obs[i].copy()
                batch_infos[i] = i_info.copy()
        else:
            batch_infos = np.tile((info[0],), batch_size)

        batch_rewards = np.tile((reward,), batch_size)
        batch_dones   = np.tile((done,), batch_size).astype(bool)

        batch_rewards = batch_rewards.reshape((batch_size, 1))
        batch_dones   = batch_dones.reshape((batch_size, 1))
        batch_infos   = batch_infos.reshape((batch_size))

        return batch_obs, batch_rewards, batch_dones, batch_infos

    def aug_test_step(self, action):
        """
            Take a single step in the environment using the given
            action, allow the environment to augment the returned
            observation. Since we're in test mode, we return a single
            instance from the batch.

            NOTE: the action is expected to be a SINGLE action. This does
            not currently support multiple environment instances.

            Arguments:
                action    The action to take.

            Returns:
                Observation, reward, done, and info (possibly augmented).
        """
        obs, reward, done, info = self.env.step(action)

        batch_obs  = self.env.augment_observation(obs)
        batch_size = batch_obs.shape[0]

        if self.test_idx < 0:
            self.test_idx = np.random.randint(0, batch_size)

        if "terminal observation" in info:
            terminal_obs = info["terminal observation"]
            terminal_obs = self.env.augment_observation(terminal_obs)
            info["terminal observation"] = terminal_obs[self.test_idx].copy()

        return batch_obs[self.test_idx], reward, done, info

    def reset(self):
        """
            Reset the environment. If we're in test mode, we don't augment the
            resulting observations. Otherwise, augment them before returning.

            Returns:
                The resulting observation(s).
        """
        if self.test_mode:
            return self.aug_test_reset()
        return self.aug_reset()

    def aug_reset(self):
        """
            Reset the environment, and return a batch of augmented observations.

            Returns:
                The resulting observations.
        """
        obs = self.env.reset()

        aug_obs_batch = self.env.augment_observation(obs)

        return aug_obs_batch

    def aug_test_reset(self):
        """
            Reset the environment, and return a single observations which may
            or may not be augmented.

            Returns:
                The resulting observation (possibly augmented).
        """
        obs = self.env.reset()

        aug_obs_batch = self.env.augment_observation(obs)
        batch_size    = aug_obs_batch.shape[0]

        if self.test_idx < 0:
            self.test_idx = np.random.randint(0, batch_size)

        return aug_obs_batch[self.test_idx]

    def get_batch_size(self):
        """
            If any wrapped classes define this method, try to get the batch size
            from them. Otherwise, assume we have a single environment.

            Returns:
                Return our batch size.
        """
        return self.batch_size

    def supports_batched_environments(self):
        """
            Determine whether or not our wrapped environment supports
            batched environments.

            Return:
                True.
        """
        return True

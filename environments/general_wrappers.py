"""
    A home for generic environment wrappers. The should not be
    specific to any type of environment.
"""
from ppo_and_friends.utils.stats import RunningMeanStd
import numpy as np
from copy import deepcopy
from functools import reduce
import pickle
import sys
import os
from abc import ABC, abstractmethod
from ppo_and_friends.utils.mpi_utils import rank_print
from ppo_and_friends.utils.misc import need_action_squeeze
from collections.abc import Iterable
from gym.spaces import Box, Discrete, MultiDiscrete, MultiBinary, Dict, Tuple
import gym
from mpi4py import MPI

comm      = MPI.COMM_WORLD
rank      = comm.Get_rank()
num_procs = comm.Get_size()

class IdentityWrapper(ABC):
    """
        A wrapper that acts exactly like the original environment but also
        has a few extra bells and whistles.
    """

    def __init__(self,
                 env,
                 test_mode = False,
                 **kw_args):
        """
            Initialize the wrapper.

            Arguments:
                env           The environment to wrap.
        """
        super(IdentityWrapper, self).__init__(**kw_args)

        self.env               = env
        self.test_mode         = test_mode
        self.need_hard_reset   = True
        self.obs_cache         = None
        self.can_augment_obs   = False

        self.observation_space        = env.observation_space
        self.critic_observation_space = env.critic_observation_space
        self.action_space             = env.action_space
        self.null_actions             = env.null_actions

        self.agent_ids = tuple(agent_id for agent_id in
            self.action_space.keys())

        if (callable(getattr(self.env, "augment_observation", None)) and
            callable(getattr(self.env, "augment_global_observation", None))):
            self.can_augment_obs = True

    def get_all_done(self):
        """
        """
        return self.env.get_all_done()

    def get_num_agents(self):
        """
            Get the number of agents in this environment.

            Returns:
                The number of agents in the environment.
        """
        return len(self.observation_space.keys())

    def set_random_seed(self, seed):
        """
            Set the random seed for the environment.

            Arguments:
                seed    The seed value.
        """
        try:
            self.env.seed(seed)
        except:
            msg  = "WARNING: unable to set the environment seed. "
            msg += "You may witness stochastic behavior."
            rank_print(msg)
            pass

        for key in self.action_space:
            self.action_space[key].seed(seed)

    def _cache_step(self, action):
        """
            Take a single step in the environment using the given
            action, and cache the observation for soft resets.

            Arguments:
                action    The action to take.

            Returns:
                The resulting observation, reward, done, and info tuple.
        """
        obs, global_obs, reward, done, info = self.env.step(action)

        self.obs_cache        = deepcopy(obs)
        self.global_obs_cache = deepcopy(global_obs)
        self.need_hard_reset  = False

        return obs, global_obs, reward, done, info

    def step(self, action):
        """
            Take a single step in the environment using the given
            action.

            Arguments:
                action    The action to take.

            Returns:
                The resulting observation, reward, done, and info tuple.
        """
        return self._cache_step(action)

    def reset(self):
        """
            Reset the environment.

            Returns:
                The resulting observation.
        """
        obs, global_state = self.env.reset()
        return obs, global_state

    def _env_soft_reset(self):
        """
            Check if our enviornment can perform a soft reset. If so, do it.
            If not, perform a standard reset.

            Returns:
                If our environment has a soft_reset method, we return the result
                of calling env.soft_reset. Otherwise, we return our obs_cache.
        """
        #
        # Tricky Business:
        # Perform a recursive call on all wrapped environments to check
        # for an ability to soft reset. Here's an example of how this works
        # in practice:
        #
        #   Imagine we wrap our environment like so:
        #   env = MultiAgentWrapper(ObservationNormalizer(ObservationClipper(env)))
        #
        #   When we call 'env.soft_reset()', we are calling
        #   ObservationClipper.soft_reset(). In this case, though, the clipper
        #   does not have direct access to the global observations from the
        #   MultiAgentWrapper. So, we recursively travel back to the base
        #   wrapper, and return its soft_reset() results. Those results will
        #   then travel back up through the normalizer and clipper, being
        #   normalized and clipped, before being returned. It's basically
        #   identical to our "step" and "reset" calls, except each wrapper
        #   may have its own cache. The takeaway is that we ignore all caches
        #   other than the bottom level cache, which is transformed on its
        #   way out of the depths of recursion.
        #
        soft_reset = getattr(self.env, "soft_reset", None)

        if callable(soft_reset):
            return soft_reset()

        return self.obs_cache, self.global_obs_cache

    def soft_reset(self):
        """
            Perform a "soft reset". This results in only performing the reset
            if the environment hasn't been reset since being created. This can
            allow us to start a new rollout from a previous rollout that ended
            near later in the environments timesteps.

            Returns:
                An observation.
        """
        if self.need_hard_reset or type(self.obs_cache) == type(None):
            return self.reset()

        return self._env_soft_reset()

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

    def augment_observation(self, obs):
        """
            If our environment has defined an observation augmentation
            method, we can access it here.

            Arguments:
                The observation to augment.

            Returns:
                The batch of augmented observations.
        """
        if self.can_augment_obs:
            return self.env.augment_observation(obs)
        else:
            raise NotImplementedError

    def augment_global_observation(self, obs):
        """
            If our environment has defined a global observation augmentation
            method, we can access it here.

            Arguments:
                The observation to augment.

            Returns:
                The batch of augmented observations.
        """
        if self.can_augment_obs:
            return self.env.augment_global_observation(obs)
        else:
            raise NotImplementedError

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

    def supports_batched_environments(self):
        """
            Determine whether or not our wrapped environment supports
            batched environments.

            Return:
                True or False.
        """
        batch_support = getattr(self.env, "supports_batched_environments", None)

        if callable(batch_support):
            return batch_support()
        else:
            return False

    def get_batch_size(self):
        """
            If any wrapped classes define this method, try to get the batch size
            from them. Otherwise, assume we have a single environment.

            Returns:
                Our batch size.
        """
        batch_size_getter = getattr(self.env, "get_batch_size", None)

        if callable(batch_size_getter):
            return batch_size_getter()
        else:
            return 1

    def has_wrapper(self, wrapper_class):
        """
            Recursively check to see if any of the wrappers around
            our environemnt match the given wrapper_class.

            Arguments:
                wrapper_class    The wrapper that we're interested in.

            Returns:
                True iff the environment is wrapped with wrapper_class.
        """
        if isinstance(self, wrapper_class):
            return True
        else:
            has_wrapper = getattr(self.env, "has_wrapper", None)

            if callable(has_wrapper):
                return has_wrapper(wrapper_class)

        return False


#FIXME: move to some other file?
class PPOEnvironmentWrapper(ABC):
    """
        A wrapper that acts exactly like the original environment but also
        has a few extra bells and whistles.
    """

    def __init__(self,
                 env,
                 test_mode         = False,
                 add_agent_ids     = False,
                 critic_view       = "global",
                 policy_mapping_fn = None,
                 **kw_args):
        """
            Initialize the wrapper.

            Arguments:
                env           The environment to wrap.
        """
        super(PPOEnvironmentWrapper, self).__init__(**kw_args)

        critic_view = critic_view.lower()
        assert critic_view in ["global", "local", "policy"]

        if critic_view == "policy" and policy_mapping_fn is None:
            msg  = "ERROR: policy_mapping_fn must be set when "
            msg += "critic_view is set to 'policy'."
            rank_print(msg)
            comm.Abort()

        self.env               = env
        self.all_done          = False
        self.null_actions      = {}
        self.add_agent_ids     = add_agent_ids
        self.critic_view       = critic_view
        self.policy_mapping_fn = policy_mapping_fn

        self.action_space             = Dict()
        self.observation_space        = Dict()
        self.critic_observation_space = Dict()

        self._define_multi_agent_spaces()
        self._define_critic_space()

        if callable(getattr(self.env, "augment_observation", None)):
            self.can_augment_obs = True

        self.agent_int_ids = {}
        for a_idx, a_id in enumerate(self.agent_ids):
            self.agent_int_ids[a_id] = a_idx

    def _apply_death_mask(self, obs, done):
        """
        """
        if np.array(done).all():
            return obs

        #
        # TODO: we may want to allow users to define what
        # values their death mask uses.
        #
        for agent_id in done:
            if done[agent_id]:
                obs[agent_id]  = np.zeros_like(obs[agent_id])
                done[agent_id] = False

        return obs

    def _add_agent_ids_to_obs(self, obs):
        """
        """
        for a_id in obs:
            obs[a_id] = np.concatenate((obs[a_id],
                (self.agent_int_ids[a_id],))).astype(obs[a_id].dtype) /\
                self.num_agents

        return obs

    def _define_critic_space(self):
        """
        """
        #
        # Local view: the critics only see what the actor sees.
        #
        if self.critic_view == "local":
            for a_id in self.agent_ids:
                self.critic_observation_space[a_id] = \
                    self.observation_space[a_id]

        #
        # Global view: the critics see everything (observations from all
        # available agents).
        #
        elif self.critic_view == "global":
            for a_id in self.agent_ids:
                self.critic_observation_space[a_id] = \
                    gym.spaces.flatten_space(self.observation_space)

        #
        # Policy view: the critics see observations from all agents that
        # share policies.
        #
        elif self.critic_view == "policy":
            self.policy_spaces = {}

            #
            # First, map policy ids to shared spaces.
            #
            for a_id in self.agent_ids:
                policy_id = self.policy_mapping_fn(a_id)

                if a_id not in self.policy_spaces:
                    self.policy_spaces[policy_id] = []

                self.policy_spaces[policy_id].append(
                    self.observation_space[a_id])

            #
            # Next, flatten the spaces so that each policy has a single space.
            #
            for policy_id in self.policy_spaces:
                self.policy_spaces[policy_id] = \
                    gym.spaces.flatten_space(Tuple(
                        self.policy_spaces[policy_id]))

            #
            # Lastly, we can assign spaces to agents.
            #
            for a_id in self.agent_ids:
                policy_id = self.policy_mapping_fn(a_id)
                self.critic_observation_space[a_id] = \
                    self.policy_spaces[policy_id]

    def _construct_global_critic_observation(self,
                                             obs,
                                             done):
        """
        """
        #
        # All agents will share the same critic observations.
        # Let's be memory sensitive here and only construct
        # one observation that all agents can share.
        #
        critic_obs = {}

        agent0 = next(iter(self.agent_ids))
        critic_shape = (1, reduce(lambda a, b: a * b,
            self.critic_observation_space[agent0].shape))

        critic_obs_data = np.zeros(critic_shape)

        critic_obs_data = critic_obs_data.astype(
            self.critic_observation_space[agent0].dtype)

        #
        # First pass: construct the shared observation.
        #
        start = 0
        for a_id in self.agent_ids:
            obs_size = reduce(lambda a, b: a * b,
                self.observation_space[a_id].shape)

            stop = start + obs_size

            #
            # There are two cases where we zero out the obervations:
            #  1. The agent has died, but we're not all done (death masking).
            #  2. An agent hasn't acted in this turn ("turn masking"). We could
            #     also keep around previous turn observations in these cases,
            #     but it's unclear if that would be useful. TODO: investigate.
            #
            if a_id not in obs or (done[a_id] and not self.all_done):
                mask = np.zeros(obs_size)
                mask = mask.astype(critic_obs_data.dtype)
                critic_obs_data[0, start : stop] = mask.flatten()

            elif a_id in obs:
                critic_obs_data[0, start : stop] = obs[a_id].flatten()

            start = stop

        #
        # Second pass: assign the shared observation.
        #
        for a_id in self.agent_ids:
            critic_obs[a_id] = critic_obs_data

        return critic_obs

    def _construct_policy_critic_observation(self,
                                             obs,
                                             done):
        """
        """
        critic_obs  = {}
        policy_data = {}

        for policy_id in self.policy_spaces:
            data = np.zeros((1, reduce(lambda a, b : a * b,
                self.policy_spaces[policy_id].shape)))

            data = critic_obs_data.astype(
                self.policy_spaces[policy_id].dtype)

            policy_data[policy_id] = {"start" : 0, "data" : data}

        #
        # First pass: construct the shared observations.
        #
        for a_id in self.agent_ids:
            policy_id = self.policy_mapping_fn(a_id)


            obs_size = reduce(lambda a, b: a * b,
                self.observation_space[a_id].shape)

            start = policy_data[policy_id]["start"]
            stop  = start + obs_size

            #
            # There are two cases where we zero out the obervations:
            #  1. The agent has died, but we're not all done (death masking).
            #  2. An agent hasn't acted in this turn ("turn masking"). We could
            #     also keep around previous turn observations in these cases,
            #     but it's unclear if that would be useful. TODO: investigate.
            #
            if a_id not in obs or (done[a_id] and not self.all_done):
                mask      = np.zeros(obs_size)
                mask      = mask.astype(critic_obs_data.dtype)
                agent_obs = mask.flatten()

            elif a_id in obs:
                agent_obs = obs[a_id].flatten()

            policy_data[policy_id]["data"][0][start : stop] = agent_obs

            policy_data[policy_id]["start"] = stop

        #
        # Second pass: assign the shared observations.
        #
        for a_id in self.agent_ids:
            policy_id        = self.policy_mapping_fn(a_id)
            critic_obs[a_id] = policy_data[policy_id]["data"]

        return critic_obs

    def _construct_critic_observation(self,
                                      obs,
                                      done):
        """
        """
        if self.critic_view == "global":
            return self._construct_global_critic_observation(obs, done)
        if self.critic_view == "local":
            return deepcopy(obs)
        if self.critic_view == "policy":
            return self._construct_policy_critic_observation(obs, done)
        else:
            rank_print(f"ERROR: unknown critic_view, {self.critic_view}.")
            comm.Abort()

    @abstractmethod
    def _define_multi_agent_spaces(self):
        """
        """
        return

    def get_all_done(self):
        """
        """
        return self.all_done

    @abstractmethod
    def step(self, action):
        """
            Take a single step in the environment using the given
            action.

            Arguments:
                action    The action to take.

            Returns:
                The resulting observation, reward, done, and info tuple.
        """
        return

    @abstractmethod
    def reset(self):
        """
            Reset the environment.

            Returns:
                The resulting observation.
        """
        return

    def render(self, **kw_args):
        """
            Render the environment.
        """
        return self.env.render(**kw_args)

    def seed(self, seed):
        """
        """
        self.env.seed(seed)


# FIXME: move to gym_wrappers?
class PPOGymWrapper(PPOEnvironmentWrapper):

    def step(self, actions):
        """
        """
        obs, global_obs, reward, done, info = \
            self._wrap_gym_step(*self.env.step(
                self._unwrap_action(actions)))

        return obs, global_obs, reward, done, info

    def reset(self):
        """
        """
        obs, global_obs = self._wrap_gym_reset(self.env.reset())
        return obs, global_obs

    @abstractmethod
    def _unwrap_action(self,
                       action):
        """
        """
        return

    @abstractmethod
    def _wrap_gym_step(self,
                       obs,
                       reward,
                       done,
                       info):
        """
        """
        return

    @abstractmethod
    def _wrap_gym_reset(self,
                        obs):
        """
        """
        return


class SingleAgentGymWrapper(PPOGymWrapper):

    def __init__(self,
                 env,
                 test_mode   = False,
                 critic_view = "local",
                 **kw_args):
        """
        """
        self.agent_ids = ("agent0",)

        super(SingleAgentGymWrapper, self).__init__(
            env,
            test_mode,
            critic_view = critic_view,
            **kw_args)

        if self.add_agent_ids:
            msg  = "WARNING: adding agent ids is not applicable "
            msg += "for single agent simulators. Disregarding."
            rank_print(msg)

        #
        # Environments are very inconsistent! Some of them require their
        # actions to be squeezed before they can be sent to the env.
        #
        self.action_squeeze = need_action_squeeze(self.env)

    def _define_multi_agent_spaces(self):
        """
        """
        for a_id in self.agent_ids:
            self.action_space[a_id]      = self.env.action_space
            self.observation_space[a_id] = self.env.observation_space

    def get_agent_id(self):
        """
        """
        if len(self.agent_ids) != 1:
            msg  = "ERROR: SingleAgentGymWrapper expects a single agnet, "
            msg += "but there are {}".format(len(self.agent_ids))
            rank_print(msg)
            comm.Abort()

        return self.agent_ids[0]

    def _unwrap_action(self,
                       action):
        """
        """
        agent_id   = self.get_agent_id()
        env_action = action[agent_id]

        if self.action_squeeze:
            env_action = env_action.squeeze()

        return env_action

    def _wrap_gym_step(self,
                       obs,
                       reward,
                       done,
                       info):
        """
        """
        agent_id = self.get_agent_id()

        #
        # HACK: some environments are buggy and don't follow their
        # own rules!
        #
        obs = obs.reshape(self.observation_space[agent_id].shape)

        if type(reward) == np.ndarray:
            reward = reward[0]

        reward = np.float32(reward)

        if done:
            self.all_done = True
        else:
            self.all_done = False

        obs      = {agent_id : obs}
        reward   = {agent_id : reward}
        done     = {agent_id : done}
        info     = {agent_id : info}

        if self.add_agent_ids:
            obs = self._add_agent_ids_to_obs(obs)

        obs        = self._apply_death_mask(obs, done)
        global_obs = self._construct_critic_observation(obs, done)

        return obs, global_obs, reward, done, info

    def _wrap_gym_reset(self,
                        obs):
        """
        """
        agent_id = self.get_agent_id()

        #
        # HACK: some environments are buggy and don't follow their
        # own rules!
        #
        obs = obs.reshape(self.observation_space[agent_id].shape)
        obs = {agent_id : obs}

        done = {agent_id : False}

        if self.add_agent_ids:
            obs = self._add_agent_ids_to_obs(obs)

        global_obs = self._construct_critic_observation(obs, done)

        return obs, global_obs


# FIXME: we need to make a note of the fact that this
# wrapper requires that all agents step at once, even if
# dead.
class MultiAgentGymWrapper(PPOGymWrapper):

    def __init__(self,
                 env,
                 test_mode     = False,
                 add_agent_ids = True,
                 **kw_args):
        """
        """
        self.num_agents = len(env.observation_space)
        self.agent_ids  = tuple(f"agent{i}" for i in range(self.num_agents))

        super(MultiAgentGymWrapper, self).__init__(
            env,
            test_mode,
            add_agent_ids = add_agent_ids,
            **kw_args)

        #
        # Environments are very inconsistent! Some of them require their
        # actions to be squeezed before they can be sent to the env.
        #
        self.action_squeeze = need_action_squeeze(self.env)

    def _define_multi_agent_spaces(self):
        """
        """
        #
        # Some gym environments are buggy and require a reshape.
        #
        self.enforced_obs_shape = {}

        for a_idx, a_id in enumerate(self.agent_ids):
            if self.add_agent_ids:
                self.observation_space[a_id] = \
                    self._expand_space_for_ids(self.env.observation_space[a_idx])
            else:
                self.observation_space[a_id] = self.env.observation_space[a_idx]

            self.enforced_obs_shape[a_id] = \
                self.env.observation_space[a_idx].shape

            self.action_space[a_id] = self.env.action_space[a_idx]

    def _expand_space_for_ids(self, space):
        """
        """
        if issubclass(type(space), Box):
            low   = space.low
            high  = space.high
            shape = space.shape

            low   = low.flatten()
            high  = high.flatten()
            shape = (reduce(lambda a, b: a * b, shape) + 1,)

            low   = np.concatenate((low, (0,)))
            high  = np.concatenate((high, (self.num_agents,)))

            return Box(
                low   = low,
                high  = high,
                shape = shape,
                dtype = space.dtype)

        elif issubclass(type(space), Discrete):
            return Discrete(space.n + 1)

    def _unwrap_action(self,
                       actions):
        """
        """
        gym_actions = np.array([None] * self.num_agents)

        for a_idx, a_id in enumerate(self.agent_ids):
            env_action = actions[a_id]

            if self.action_squeeze:
                env_action = env_action.squeeze()

            gym_actions[a_idx] = env_action

        return tuple(gym_actions)

    def _wrap_gym_step(self,
                       obs,
                       reward,
                       done,
                       info):
        """
        """
        wrapped_obs    = {}
        wrapped_reward = {}
        wrapped_done   = {}
        wrapped_info   = {}

        for a_idx, a_id in enumerate(self.agent_ids):
            agent_obs    = obs[a_idx]
            agent_reward = reward[a_idx]
            agent_done   = done[a_idx]
            agent_info   = info

            #
            # HACK: some environments are buggy and don't follow their
            # own rules!
            #
            agent_obs = agent_obs.reshape(self.enforced_obs_shape[a_id])

            if type(agent_reward) == np.ndarray:
                agent_reward = agent_reward[0]

            agent_reward = np.float32(agent_reward)

            wrapped_obs[a_id]    = agent_obs
            wrapped_reward[a_id] = agent_reward
            wrapped_info[a_id]   = agent_info
            wrapped_done[a_id]   = agent_done

        if np.array(done).all():
            self.all_done = True
        else:
            self.all_done = False

        if self.add_agent_ids:
            wrapped_obs = self._add_agent_ids_to_obs(wrapped_obs)

        obs        = self._apply_death_mask(obs, done)
        global_obs = self._construct_critic_observation(
            wrapped_obs, wrapped_done)

        return (wrapped_obs, global_obs,
            wrapped_reward, wrapped_done, wrapped_info)

    def _wrap_gym_reset(self,
                        obs):
        """
        """
        wrapped_obs  = {}
        wrapped_done = {}

        for a_idx, a_id in enumerate(self.agent_ids):
            agent_obs = obs[a_idx]

            #
            # HACK: some environments are buggy and don't follow their
            # own rules!
            #
            agent_obs = agent_obs.reshape(self.enforced_obs_shape[a_id])
            wrapped_obs[a_id] = agent_obs

            wrapped_done[a_id] = False

        if self.add_agent_ids:
            wrapped_obs = self._add_agent_ids_to_obs(wrapped_obs)

        global_obs = self._construct_critic_observation(
            wrapped_obs, wrapped_done)

        return wrapped_obs, global_obs


class VectorizedEnv(IdentityWrapper, Iterable):
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
                 env_generator,
                 num_envs = 1,
                 **kw_args):
        """
            Initialize our vectorized environment.

            Arguments:
                env_generator    A function that creates instances of our
                                 environment.
                num_envs         The number of environments to maintain.
        """
        super(VectorizedEnv, self).__init__(
            env_generator(),
            **kw_args)

        self.num_envs = num_envs
        self.envs     = np.array([None] * self.num_envs, dtype=object)
        self.iter_idx = 0

        if self.num_envs == 1:
            self.envs[0] = self.env
        else:
            for i in range(self.num_envs):
                self.envs[i] = env_generator()

    def set_random_seed(self, seed):
        """
            Set the random seed for the environment.

            Arguments:
                seed    The seed value.
        """
        for env_idx in range(self.num_envs):
            self.envs[env_idx].seed(seed)
            self.envs[env_idx].action_space.seed(seed)

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
        #
        # If we're testing, we don't want to return a batch.
        #
        if self.test_mode:
            return self.single_step(action)

        return self.batch_step(action)


    def single_step(self, action):
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
        obs, global_obs, reward, done, info = self.env.step(action)

        if self.env.get_all_done():
            for agent_id in info:
                #FIXME: deepcopy  might not be needed here.
                info[agent_id]["terminal observation"] = deepcopy(obs[agent_id])

            obs, global_obs = self.env.reset()

        return obs, global_obs, reward, done, info

    #FIXME: need to integrate this with MAPPO somehow. Each environment instance
    # will have the same agents.
    # When calling step(), we'll expect to get the standard tuple of obs, reward, etc.
    # but each one will be a dictionary mapping agent ids to their obs, reward, etc.
    # For vectorized environemnts, could we just return arrays of dictionaries?
    # We could always vectorize (which is what we did in the non-mappo case), which
    # would result in an array of size 1 when num_envs == 1. This would homogenize
    # things.
    def batch_step(self, actions):
        """
            Take a step in our environment with the given actions.
            Since we're vectorized, we reset the environment when
            we've reached a "done" state, and we return a batch
            of step results. Since this is a batch step, we'll return
            a batch of results of size self.num_envs.

            Arguments:
                actions    The actions to take.

            Returns:
                The resulting observation, reward, done, and info
                tuple.
        """
        batch_obs        = {}
        batch_global_obs = {}
        batch_rewards    = {}
        batch_dones      = {}
        batch_infos      = {}

        #
        # Each agent keeps track of its own batches.
        #
        for agent_id in self.agent_ids:
            obs_shape = (self.num_envs,) + \
                self.observation_space[agent_id].shape

            global_obs_shape = \
                (self.num_envs,) + \
                 self.critic_observation_space[agent_id].shape

            batch_obs[agent_id]        = np.zeros(obs_shape)
            batch_global_obs[agent_id] = np.zeros(global_obs_shape)
            batch_rewards[agent_id] = np.zeros((self.num_envs, 1))
            batch_dones[agent_id]   = np.zeros((self.num_envs, 1)).astype(bool)
            batch_infos[agent_id]   = np.array([None] * self.num_envs)

        env_actions = np.array([{}] * self.num_envs)
        for agent_id in actions:
            for b_idx in range(self.num_envs):
                env_actions[b_idx][agent_id] = actions[agent_id][b_idx]

        for env_idx in range(self.num_envs):
            act = env_actions[env_idx]

            obs, global_obs, reward, done, info = self.envs[env_idx].step(act)

            if self.envs[env_idx].get_all_done():
                for agent_id in info:
                    info[agent_id]["terminal observation"] = \
                        deepcopy(obs[agent_id])

                obs, global_obs = self.envs[env_idx].reset()

            for agent_id in obs:
                batch_obs[agent_id][env_idx]        = obs[agent_id]
                batch_global_obs[agent_id][env_idx] = global_obs[agent_id]
                batch_rewards[agent_id][env_idx]    = reward[agent_id]
                batch_dones[agent_id][env_idx]      = done[agent_id]
                batch_infos[agent_id][env_idx]      = info[agent_id]

        self.obs_cache = deepcopy(batch_obs)
        self.global_obs_cache = deepcopy(batch_global_obs)
        self.need_hard_reset = False

        return (batch_obs, batch_global_obs,
            batch_rewards, batch_dones, batch_infos)

    def reset(self):
        """
            Reset the environment.

            Returns:
                The resulting observation.
        """
        if self.test_mode:
            return self.single_reset()
        return self.batch_reset()

    def soft_reset(self):
        """
        """
        if self.need_hard_reset:
            return self.reset()
        return self.obs_cache, self.global_obs_cache

    def single_reset(self):
        """
            Reset the environment.

            Returns:
                The resulting observation.
        """
        obs, global_obs = self.env.reset()
        return obs, global_obs

    def batch_reset(self):
        """
            Reset the batch of environments.

            Returns:
                The resulting observation.
        """
        batch_obs        = {}
        batch_global_obs = {}

        for env_idx in range(self.num_envs):
            obs, global_obs = self.envs[env_idx].reset()

            for agent_id in obs:
                if agent_id not in batch_obs:
                    obs_shape = (self.num_envs,) + \
                        self.observation_space[agent_id].shape
                    global_obs_shape = \
                        (self.num_envs,) + \
                         self.critic_observation_space[agent_id].shape

                    batch_obs[agent_id] = np.zeros(obs_shape).astype(np.float32)
                    batch_global_obs[agent_id] = \
                        np.zeros(global_obs_shape).astype(np.float32)

                batch_obs[agent_id][env_idx]        = obs[agent_id]
                batch_global_obs[agent_id][env_idx] = global_obs[agent_id]

        return batch_obs, batch_global_obs

    def __len__(self):
        """
            Represent our length as our batch size.

            Returns:
                The number of environments in our batch.
        """
        return self.num_envs

    def __next__(self):
        """
            Allow iteration through our array of environments.

            Returns:
                The next environment in our array.
        """
        if self.iter_idx < self.num_envs:
            env = self.envs[self.iter_idx]
            self.iter_idx += 1
            return env

        raise StopIteration

    def __iter__(self):
        """
            Allow iteration through our array of environments.

            Returns:
                Ourself as an iterable.
        """
        return self

    def __getitem__(self, idx):
        """
            Allow accessing environments by index.

            Arguments:
                idx    The index to the desired environment.

            Returns:
                The environment from self.envs located at index idx.
        """
        return self.envs[idx]

    def supports_batched_environments(self):
        """
            Determine whether or not our wrapped environment supports
            batched environments.

            Return:
                True or False.
        """
        return True

    def get_batch_size(self):
        """
            Get the batch size.

            Returns:
                Return our batch size.
        """
        return len(self)

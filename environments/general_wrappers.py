"""
    A home for generic environment wrappers. The should not be
    specific to any type of environment.
"""
from ppo_and_friends.utils.stats import RunningMeanStd
import numpy as np
from copy import deepcopy
import pickle
import sys
import os
from abc import ABC, abstractmethod
from ppo_and_friends.utils.mpi_utils import rank_print
from ppo_and_friends.utils.misc import need_action_squeeze
from collections.abc import Iterable
from gym.spaces import Box, Discrete, MultiDiscrete, MultiBinary, Dict
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
                status_dict   The dictionary containing training stats.
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

        self.agent_ids = {agent_id for agent_id in self.action_space.keys()}

        if callable(getattr(self.env, "augment_observation", None)):
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

    #FIXME: we should probably rename global_state to global_obs
    def get_global_state_space(self):
        """
            Get the global state space. This is specific to multi-agent
            environments. In single agent environments, it is the same
            as the observation_space.

            Returns:
                If available, the global state space is returned. Otherwise,
                the observation space is returned.
        """
        return self.critic_observation_space

    def get_global_state(self, obs):
        """
            From arXiv:2103.01955v2, cacluate the "Feature-Pruned
            Agent-Specific Global State". By default, we assume that
            the environment doesn't offer extra information, which leads
            us to actually caculating the "Concatenation of Local Observations".
            We can create sub-classes of this implementation to handle
            environments that do offer global state info.

            Arguments:
                obs    The refined agent observations.

            Returns:
                The global state to be fed to the critic.
        """
        #
        # If our wrapped environment has its own implementation, rely
        # on that. Otherwise, we'll just concatenate all of the agents
        # together.
        #
        get_global_state = getattr(self.env,
            "get_global_state", None)

        if callable(get_global_state):
            return get_global_state(obs)

        #
        # FIXME: make sure we don't hit memory limits here.
        #
        global_state   = {}
        all_agents_obs = np.array([None] * self.get_num_agents())

        for idx, agent_id in enumerate(obs):
            all_agents_obs[idx] = obs[agent_id]

        all_agents_obs = np.concatenate(all_agents_obs).flatten()
        all_agents_obs = all_agents_obs.astype(np.float32)

        for agent_id in obs:
            global_state[agent_id] = all_agents_obs

        return global_state

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
        obs, reward, done, info = self.env.step(action)

        self.obs_cache = deepcopy(obs)
        self.need_hard_reset = False

        global_state = self.get_global_state(obs)

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

        global_state = self.get_global_state(self.obs_cache)

        return self.obs_cache, global_state

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
                 test_mode = False,
                 **kw_args):
        """
            Initialize the wrapper.

            Arguments:
                env           The environment to wrap.
                status_dict   The dictionary containing training stats.
        """
        super(PPOEnvironmentWrapper, self).__init__(**kw_args)

        self.env          = env
        self.all_done     = False
        self.null_actions = {}

        #FIXME: these will need to be converted to dictionaries if they're not alread.
        # we also need obs spaces for actors and critics.
        #self.observation_space = env.observation_space
        #self.action_space      = env.action_space
        self.define_multi_agent_spaces()

        if callable(getattr(self.env, "augment_observation", None)):
            self.can_augment_obs = True

    @abstractmethod
    def define_multi_agent_spaces(self):
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


# FIXME: will the environment generators need to be wrapped by
# these multi-agent interfaces now?? We could then pass that
# generator to the vectorized wrapper. This probably makes
# the most sense...
#
# FIXME: move to gym_wrappers?
class SingleAgentGymWrapper(PPOEnvironmentWrapper):

    def __init__(self,
                 env,
                 test_mode = False,
                 **kw_args):
        """
        """
        self.agent_ids = {"agent0"}

        super(SingleAgentGymWrapper, self).__init__(
            env,
            test_mode,
            **kw_args)
        #
        # Environments are very inconsistent! Some of them require their
        # actions to be squeezed before they can be sent to the env.
        #
        self.action_squeeze = need_action_squeeze(self.env)

    def define_multi_agent_spaces(self):
        """
        """
        self.action_space             = Dict()
        self.observation_space        = Dict()
        self.critic_observation_space = Dict()

        for a_id in self.agent_ids:
            self.action_space[a_id]             = self.env.action_space
            self.observation_space[a_id]        = self.env.observation_space
            self.critic_observation_space[a_id] = self.env.observation_space

    def get_agent_id(self):
        """
        """
        if len(self.agent_ids) != 1:
            msg  = "ERROR: SingleAgentGymWrapper expects a single agnet, "
            msg += "but there are {}".format(len(self.agent_ids))
            rank_print(msg)
            comm.Abort()

        return tuple(self.agent_ids)[0]

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
        obs = obs.reshape(self.observation_space.shape)

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

        return obs, deepcopy(obs), reward, done, info

    def _wrap_gym_reset(self,
                        obs):
        """
        """
        agent_id = self.get_agent_id()

        #
        # HACK: some environments are buggy and don't follow their
        # own rules!
        #
        obs = obs.reshape(self.observation_space.shape)
        obs = {agent_id : obs}

        return obs, deepcopy(obs)

    def step(self, actions):
        """
        """
        obs, global_obs, reward, done, info = \
            self._wrap_gym_step(*self.env.step(
                self._unwrap_action(actions)))

        self.obs_cache = deepcopy(obs)
        self.need_hard_reset = False

        return obs, global_obs, reward, done, info

    def reset(self):
        """
        """
        obs, global_obs = self._wrap_gym_reset(self.env.reset())
        return obs, global_obs

    #FIXME: cleanup
    #def soft_reset(self):
    #    """
    #    """
    #    if self.need_hard_reset:
    #        return self.reset()
    #    return self.obs_cache, self.obs_cache

    def seed(self, seed):
        """
        """
        self.env.seed(seed)


#FIXME need to handle dicts
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

    #FIXME: add some of the multi-agent options (add ids, etc.)
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

        # FIXME: need to handle when not all agents are acting.
        # we also need to handle when an agent dies, but the sim is still going.
        # will this be handled in the ppo env wrapper? Probably the best bet...
        # maybe it's also best to handle non-acting agents in that same wrapper.
        # also, we should always death mask. Why would we ever want to not?
        for env_idx in range(self.num_envs):
            #act = actions[env_idx]
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

        #FIXME: integrate soft resets into this class
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
        #batch_obs        = np.array([None] * self.num_envs)
        #batch_global_obs = np.array([None] * self.num_envs)

        #for env_idx in range(self.num_envs):
        #    obs, global_obs           = self.envs[env_idx].reset()
        #    batch_obs[env_idx]        = obs
        #    batch_global_obs[env_idx] = obs

        #return batch_obs, batch_global_obs

        batch_obs        = {}
        batch_global_obs = {}

        #FIXME: need to handle when not all agents are acting.
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


# FIXME: need to handle dictionaries mapping agents to
# spaces. Also, the done needs an "all" key.
#
# This class should probably only be used for classic Gym
# environments.
#
# For Abmarl, we will want to have a simpler wrapper that
# can handle things like global observations. I think
# Abmarl's wrapper inherits from RLlib, so we can check
# for that.
#
# Is it possible for agents sharing the same policy to have
# different spaces? I don't think so... Even if they're on
# the same team, they will need separate policies.
# NOTE: a policies global observation CAN contain mixed
# spaces, so flatten_space will need to be used there.
#
#class MultiAgentWrapper(IdentityWrapper):
#    """
#        A wrapper for multi-agent environments. This design follows the
#        suggestions outlined in arXiv:2103.01955v2.
#    """
#
#    def __init__(self,
#                 env_generator,
#                 need_agent_ids   = True,
#                 use_global_state = True,
#                 normalize_ids    = True,
#                 death_mask       = True,
#                 **kw_args):
#        """
#            Initialize the wrapper.
#
#            Arguments:
#                env               The environment to wrap.
#                need_agent_ids    Do we need to explicitly add the agent
#                                  ids to their observations? Assume yes.
#                use_global_state  Send a global state to the critic.
#                normalize_ids     If we're explicitly adding ids, should
#                                  we first normalize them?
#                death_mask        Should we perform death masking?
#        """
#        super(MultiAgentWrapper, self).__init__(
#            env_generator(),
#            **kw_args)
#
#        self.need_agent_ids = need_agent_ids
#        self.num_agents     = len(self.env.observation_space)
#        self.agent_ids      = np.arange(self.num_agents).reshape((-1, 1))
#
#        if normalize_ids:
#            self.agent_ids = self.agent_ids / self.num_agents
#
#        self.use_global_state   = use_global_state
#        self.global_state_space = None
#        self.death_mask         = death_mask
#
#        # FIXME: I think our observation and action spaces will need to
#        # be dictionaries mapping agents to their spaces.
#        self.observation_space, _ = self._get_refined_space(
#            multi_agent_space = self.env.observation_space,
#            add_ids           = need_agent_ids)
#
#        self.action_space, self.num_actions = self._get_refined_space(
#            multi_agent_space = self.env.action_space,
#            add_ids           = False)
#
#        self._construct_global_state_space()
#
#    def get_num_agents(self):
#        """
#            Get the number of agents in this environment.
#
#            Returns:
#                The number of agents in the environment.
#        """
#        return self.num_agents
#
#    def get_global_state_space(self):
#        """
#            Get the global state space.
#
#            Returns:
#                self.global_state_space
#        """
#        return self.global_state_space
#
#    # FIXME: we will still need this for the global observation
#    # space, but I think we should probably use the class that
#    # Abmarl uses to combine these things. It's a Gym function.
#    # => gym.spaces.flatten_space. Note that the agent spaces dict
#    # must be a gym Dict, which is ordered. Also, the flatten_space
#    # function might have a bug with discrete spaces (see Ephraim's
#    # github issue).
#    def _get_refined_space(self,
#                           multi_agent_space,
#                           add_ids = False):
#        """
#            Given a multi-agent space (action, observation, etc.) consisting
#            of tuples of spaces, construct a single space to represent the
#            entire environment.
#
#            Some things to note:
#              1. In environmens with mixed sized spaces, it's assumed that we
#                 will be zero padding the lesser sized spaces. Therefore, the
#                 returned space will match the maximum sized space.
#              2. Mixing space types is not currently supported.
#              3. Discrete and Box are the only two currently supported spaces.
#
#            Arguments:
#                multi_agent_space    The reference space. This should consist
#                                     of tuples of spaces.
#                add_ids              Will the agent ids be appended to this
#                                     space? If so, expand the size to account
#                                     for this.
#
#            Returns:
#                A tuple containing a single space representing all agents as
#                the first element and the size of the space as the second
#                element.
#        """
#        #
#        # First, ensure that each agent has the same type of space.
#        #
#        space_type      = None
#        prev_space_type = None
#        dtype           = None
#        prev_dtype      = None
#        for space in multi_agent_space:
#            if space_type == None:
#                space_type      = type(space)
#                prev_space_type = type(space)
#
#                dtype         = space.dtype
#                prev_dtype    = space.dtype
#
#            if prev_space_type != space_type:
#                msg  = "ERROR: mixed space types in multi-agent "
#                msg += "environments is not currently supported. "
#                msg += "Found types "
#                msg += "{} and {}.".format(space_type, prev_space_type)
#                rank_print(msg)
#                comm.Abort()
#            elif prev_dtype != dtype:
#                msg  = "ERROR: mixed space dtypes in multi-agent "
#                msg += "environments is not currently supported. "
#                msg += "Found types "
#                msg += "{} and {}.".format(prev_dtype, dtype)
#                rank_print(msg)
#                comm.Abort()
#            else:
#                prev_space_type = space_type
#                space_type      = type(space)
#
#                prev_dtype    = dtype
#                dtype         = space.dtype
#
#        #
#        # Next, we need handle cases where agents have different spaces
#        # We want to homogenize things, so we pad all of the spaces
#        # to be identical.
#        #
#        # TODO: we're assuming all dimensions are flattened. In the future,
#        # we may want to support multi-dimensional spaces.
#        #
#        low   = np.empty(0, dtype = np.float32)
#        high  = np.empty(0, dtype = np.float32)
#        count = 0
#
#        for space in multi_agent_space:
#            if space_type == Box:
#                diff  = space.shape[0] - low.size
#                count = max(count, space.shape[0])
#
#                low   = np.pad(low, (0, diff))
#                high  = np.pad(high, (0, diff))
#
#                high  = np.maximum(high, space.high)
#                low   = np.minimum(low, space.low)
#
#            elif space_type == Discrete:
#                count = max(count, space.n)
#
#            else:
#                not_supported_msg  = "ERROR: {} is not currently supported "
#                not_supported_msg += "as an observation space in multi-agent "
#                not_supported_msg += "environments."
#                rank_print(not_supported_msg.format(space))
#                comm.Abort()
#
#        if space_type == Box:
#            if add_ids:
#                low   = np.concatenate((low, (0,)))
#                high  = np.concatenate((high, (np.inf,)))
#                size  = count + 1
#                shape = (count + 1,)
#            else:
#                size  = count
#
#            shape = (size,)
#
#            new_space = Box(
#                low   = low,
#                high  = high,
#                shape = (size,),
#                dtype = dtype)
#
#        elif space_type == Discrete:
#            size = 1
#            if add_ids:
#                new_space = Discrete(count + 1)
#            else:
#                new_space = Discrete(count)
#
#        return (new_space, size)
#
#    # FIXME: how do we represent the global state space if we have
#    # multiple agents with different obs spaces?
#    # use gym.spaces.flatten_space
#    # i.e. we won't need this method anymore.
#    def _construct_global_state_space(self):
#        """
#            Construct the global state space. See arXiv:2103.01955v2.
#            By deafult, we concatenate all local observations to represent
#            the global state.
#        """
#        #
#        # If we're not using the global state space approach, each agent's
#        # critic update receives its own observations only.
#        #
#        if not self.use_global_state:
#            self.global_state_space = self.observation_space
#            return
#
#        elif hasattr(self.env, "global_state_space"):
#            self.global_state_space = self.env.global_state_space
#            return
#
#        obs_type = type(self.observation_space)
#
#        if obs_type == Box:
#            low   = np.tile(self.observation_space.low, self.num_agents)
#            high  = np.tile(self.observation_space.high, self.num_agents)
#            shape = (self.observation_space.shape[0] * self.num_agents,)
#
#            global_state_space = Box(
#                low   = low,
#                high  = high,
#                shape = shape,
#                dtype = self.observation_space.dtype)
#
#        elif obs_type == Discrete:
#            global_state_space = Discrete(
#                self.observation_space.n * self.num_agents)
#        else:
#            not_supported_msg  = "ERROR: {} is not currently supported "
#            not_supported_msg += "as an observation space in multi-agent "
#            not_supported_msg += "environments."
#            rank_print(not_supported_msg.format(self.observation_space))
#            comm.Abort()
#
#        self.global_state_space = global_state_space
#
#    def get_feature_pruned_global_state(self, obs):
#        """
#            From arXiv:2103.01955v2, cacluate the "Feature-Pruned
#            Agent-Specific Global State". By default, we assume that
#            the environment doesn't offer extra information, which leads
#            us to actually caculating the "Concatenation of Local Observations".
#            We can create sub-classes of this implementation to handle
#            environments that do offer global state info.
#
#            Arguments:
#                obs    The refined agent observations.
#
#            Returns:
#                The global state to be fed to the critic.
#        """
#        #
#        # If we're not using the global state space approach, each agent's
#        # critic update receives its own observations only.
#        #
#        if not self.use_global_state:
#            return obs
#
#        #
#        # If our wrapped environment has its own implementation, rely
#        # on that. Otherwise, we'll just concatenate all of the agents
#        # together.
#        #
#        pruned_from_env = getattr(self.env,
#            "get_feature_pruned_global_state", None)
#
#        if callable(pruned_from_env):
#            return pruned_from_env(obs)
#
#        #
#        # Our observation is currently in the shape (num_agents, obs), so
#        # it's really a batch of observtaions. The global state is really
#        # just this batch concatenated as a single observation. Each agent
#        # needs a copy of it, so we just tile them. Instead of actually making
#        # copies, we can use broadcast_to to create references to the same
#        # memory location.
#        #
#        global_state = obs.flatten()
#        obs_size     = global_state.size
#        global_state = np.broadcast_to(global_state, (self.num_agents, obs_size))
#        return global_state
#
#    def _refine_obs(self, obs):
#        """
#            Refine our multi-agent observations.
#
#            Arguments:
#                The multi-agent observations.
#
#            Returns:
#                The refined multi-agent observations.
#        """
#        obs = np.stack(obs, axis=0)
#
#        if self.need_agent_ids:
#            obs = np.concatenate((self.agent_ids, obs), axis=1)
#
#        #
#        # It's possible for agents to have differing observation space sizes.
#        # In those cases, we zero pad for congruity.
#        #
#        size_diff = self.observation_space.shape[0] - obs.shape[-1]
#        if size_diff > 0:
#            obs = np.pad(obs, ((0, 0), (0, size_diff)))
#
#        return obs
#
#    def _refine_dones(self, agents_done, obs):
#        """
#            Refine our done array. Also, perform death masking when
#            appropriate. See arXiv:2103.01955v2 for information on
#            death masking.
#
#            Arguments:
#                agents_done    The done array.
#                obs            The agent observations.
#
#            Returns:
#                The refined done array as well as a single boolean
#                representing whether or not the entire environment is
#                done.
#        """
#        agents_done = np.array(agents_done)
#
#        #
#        # We assume that our environment is done only when all agents
#        # are done. If death masking is enabled and some but not all
#        # agents have died, we need to apply the mask.
#        #
#        all_done = False
#        if agents_done.all():
#            all_done = True
#        elif self.death_mask:
#            obs[agents_done, 1:] = 0.0
#            agents_done = np.zeros(self.num_agents).astype(bool)
#
#        agents_done = agents_done.reshape((-1, 1))
#        return agents_done, all_done
#
#    def step(self, actions):
#        """
#            Take a step in our multi-agent environment.
#
#            Arguments:
#                actions    The actions for each agent to take.
#
#            Returns:
#                observations, rewards, dones, and info.
#        """
#        #
#        # It's possible for agents to have differing action space sizes.
#        # In those cases, we zero pad for congruity.
#        #
#        size_diff = self.num_actions - int(actions.size / self.num_agents)
#        if size_diff > 0:
#            actions = np.pad(actions, ((0, 0), (0, size_diff)))
#
#        #
#        # Our first/test environment expects the actions to be contained in
#        # a tuple. We may need to support more formats in the future.
#        #
#        tup_actions = tuple(actions[i] for i in range(self.num_agents))
#
#        obs, rewards, agents_done, info = self.env.step(actions)
#
#        #
#        # Info can come back int two forms:
#        #   1. It's a dictionary containing global information.
#        #   2. It's an iterable containing num_agent dictionaries,
#        #      each of which contains info for its assocated agent.
#        #
#        info_is_global = False
#        if type(info) == dict:
#            info_is_global = True
#        else:
#            info = np.array(info).astype(object)
#
#        obs     = self._refine_obs(obs)
#        rewards = np.stack(np.array(rewards), axis=0).reshape((-1, 1))
#        dones, all_done = self._refine_dones(agents_done, obs)
#
#        if all_done:
#            terminal_obs = deepcopy(obs)
#            obs, global_state = self.reset()
#
#            if info_is_global:
#                info["global state"] = global_state
#            else:
#                for i in range(info.size):
#                    info[i]["global state"] = global_state
#
#        else:
#            global_state = self.get_feature_pruned_global_state(obs)
#            if info_is_global:
#                info["global state"] = global_state
#            else:
#                for i in range(info.size):
#                    info[i]["global state"] = global_state
#
#        #
#        # If our info is global, we need to convert it to local.
#        # Create an array of references so that we don't use up memory.
#        #
#        if info_is_global:
#            info = np.array([info] * self.num_agents, dtype=object)
#
#        #
#        # Lastly, each agent needs its own terminal observation.
#        #
#        if all_done:
#            for i in range(self.num_agents):
#                info[i]["terminal observation"] = deepcopy(terminal_obs[i])
#
#        elif not self.death_mask:
#            where_done = np.where(dones)[0]
#
#            for d_idx in where_done:
#                info[d_idx]["terminal observation"] = \
#                   deepcopy(obs[d_idx])
#
#        self.obs_cache = deepcopy(obs)
#        self.need_hard_reset = False
#
#        return obs, rewards, dones, info
#
#    def reset(self):
#        """
#            Reset the environment. If we're in test mode, we don't augment the
#            resulting observations. Otherwise, augment them before returning.
#
#            Returns:
#                The local and global observations.
#        """
#        obs = self.env.reset()
#        obs = self._refine_obs(obs)
#
#        global_state = self.get_feature_pruned_global_state(obs)
#
#        return obs, global_state
#
#    def soft_reset(self):
#        """
#            Perform a "soft reset". This results in only performing the reset
#            if the environment hasn't been reset since being created. This can
#            allow us to start a new rollout from a previous rollout that ended
#            near later in the environments timesteps.
#
#            Returns:
#                The local and global observations.
#        """
#        if self.need_hard_reset or type(self.obs_cache) == type(None):
#            return self.reset()
#
#        global_state = self.get_feature_pruned_global_state(self.obs_cache)
#
#        return self.obs_cache, global_state
#
#    def get_batch_size(self):
#        """
#            Get the batch size.
#
#            Returns:
#                Return our batch size.
#        """
#        return self.num_agents
#
#    def supports_batched_environments(self):
#        """
#            Determine whether or not our wrapped environment supports
#            batched environments.
#
#            Return:
#                True.
#        """
#        return True

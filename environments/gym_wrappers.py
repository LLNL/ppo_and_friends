"""
    This module contains wrappers for gym environments.
"""
from ppo_and_friends.environments.ppo_env_wrappers import PPOEnvironmentWrapper
import numpy as np
from ppo_and_friends.utils.misc import need_action_squeeze
from gym.spaces import Box, Discrete
from ppo_and_friends.utils.mpi_utils import rank_print
from abc import abstractmethod
from functools import reduce

from mpi4py import MPI
comm      = MPI.COMM_WORLD
rank      = comm.Get_rank()
num_procs = comm.Get_size()


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

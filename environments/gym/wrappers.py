"""
    This module contains wrappers for gym environments.
"""
from ppo_and_friends.environments.ppo_env_wrappers import PPOEnvironmentWrapper
import numpy as np
from gymnasium.spaces import Box, Discrete
from ppo_and_friends.utils.mpi_utils import rank_print
from abc import abstractmethod
from functools import reduce

from mpi4py import MPI
comm      = MPI.COMM_WORLD
rank      = comm.Get_rank()
num_procs = comm.Get_size()


class PPOGymWrapper(PPOEnvironmentWrapper):
    """
        OpenAI gym environments typically return numpy arrays
        for each step. This wrapper will convert these environments
        into a more multi-agent friendly setup, where each step/reset
        returns dictionaries mapping agent ids to their attributes.
        This will also return a critic observation along with the
        actor observation.
    """
    def __init__(self,
                 *args,
                 **kw_args):

        if "critic_view" in kw_args:
            supported_views = ["local", "global"]

            if kw_args["critic_view"] not in supported_views:
                msg  = "ERROR: PPO-AF gym wrappers only support critic views "
                msg += f"{supported_views}, but received "
                msg += f"{kw_args['critic_view']}."
                rank_print(msg)
                comm.Abort()

        super(PPOGymWrapper, self).__init__(
            *args,
            **kw_args)

        self.random_seed = None

    def step(self, actions):
        """
            Take a step in the environment.

            Arguments:
                actions    A dictionary mapping agent ids to actions.

            Returns:
                The observation, critic_observation, reward, done,
                and info tuple.
        """
        actions = self._filter_done_agent_actions(actions)

        obs, critic_obs, reward, terminated, truncated, info = \
            self._wrap_gym_step(*self.env.step(
                self._unwrap_action(actions)))

        return obs, critic_obs, reward, terminated, truncated, info

    def reset(self):
        """
            Reset the environment.

            Returns:
                The actor and critic observations.
        """
        obs, critic_obs = self._wrap_gym_reset(
            *self.env.reset(seed = self.random_seed))

        self._reset_step_restrictions()

        #
        # Gym versions >= 0.26 require the random seed to be set
        # when calling reset. Since we don't want the same exact
        # episode to reply every time we reset, we increment the
        # seed. This retains reproducibility while allow each episode
        # to vary.
        #
        if self.random_seed != None:
            self.random_seed += 1

        return obs, critic_obs

    @abstractmethod
    def _unwrap_action(self,
                       action):
        """
            An abstract method defining how to unwrap an action.

            Arguments:
                A dictionary mapping agent ids to actions.

            Returns:
                Agent actions that the underlying environment can
                process.
        """
        return

    @abstractmethod
    def _wrap_gym_step(self,
                       obs,
                       reward,
                       terminated,
                       truncated,
                       info):
        """
            An abstract method defining how to wrap our enviornment
            step.

            Arguments:
                obs         The agent observations.
                reward      The agent rewards.
                terminated  The agent termination flags.
                truncated   The agent truncated flags.
                info        The agent info.

            Returns:
                A tuple of form (obs, critic_obs, reward,
                terminated, truncated, info) s.t. each is a dictionary.
        """
        return

    @abstractmethod
    def _wrap_gym_reset(self,
                        obs,
                        info):
        """
            An abstract method defining how to wrap our enviornment
            reset.

            Arguments:
                obs        The agent observations.
                info       An info dictionary.

            Returns:
                A tuple of form (obs, critic_obs) s.t.
                each is a dictionary.
        """
        return

    def seed(self,
             seed):
        """
            Set the seed for this environment.

            Arguments:
                seed    The random seed.
        """
        if seed != None:
            assert type(seed) == int

        self.random_seed = seed


class SingleAgentGymWrapper(PPOGymWrapper):
    """
        A wrapper for single agent gym environments.
    """

    def __init__(self,
                 env,
                 test_mode   = False,
                 **kw_args):
        """
            Arguments:
                env            The gym environment to wrap.
                test_mode      Are we testing?
        """
        super(SingleAgentGymWrapper, self).__init__(
            env,
            test_mode,
            critic_view = "local",
            **kw_args)

        if self.add_agent_ids:
            msg  = "WARNING: adding agent ids is not applicable "
            msg += "for single agent simulators. Disregarding."
            rank_print(msg)

    def _define_agent_ids(self):
        """
            Define our agent_ids.
        """
        self.agent_ids  = ("agent0",)
        self.num_agents = 1

    def _define_multi_agent_spaces(self):
        """
            Define our multi-agent spaces. We have a single agent here,
        """
        for a_id in self.agent_ids:
            self.action_space[a_id]      = self.env.action_space
            self.observation_space[a_id] = self.env.observation_space

    def get_agent_id(self):
        """
            Get our only agent's id.

            Returns:
                Our agent's id.
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
            An method defining how to unwrap an action.

            Arguments:
                A dictionary mapping agent ids to actions.

            Returns:
                A numpy array of actions.
        """
        agent_id   = self.get_agent_id()
        env_action = action[agent_id]
        env_action = env_action.reshape(self.action_space[agent_id].shape)
        return env_action

    def _wrap_gym_step(self,
                       obs,
                       reward,
                       terminated,
                       truncated,
                       info):
        """
            A method defining how to wrap our enviornment
            step.

            Arguments:
                obs         The agent observations.
                reward      The agent rewards.
                terminated  The agent termination flags.
                truncated   The agent truncated flags.
                info        The agent info.

            Returns:
                A tuple of form (obs, critic_obs, reward,
                terminated, truncated, info) s.t. each is a dictionary.
        """
        agent_id = self.get_agent_id()

        #
        # HACK: some environments are buggy and don't follow their
        # own rules!
        #
        obs = obs.reshape(self.observation_space[agent_id].shape)

        if type(reward) == np.ndarray:
            reward = reward[0]

        terminated = {agent_id : terminated}
        truncated  = {agent_id : truncated}

        terminated, truncated = self._apply_step_restrictions(
            terminated, truncated)

        if terminated or truncated:
            self.all_done = True
        else:
            self.all_done = False

        obs        = {agent_id : obs}
        reward     = {agent_id : np.float32(reward)}
        info       = {agent_id : info}

        if self.add_agent_ids:
            obs = self._add_agent_ids_to_obs(obs)

        critic_obs = self._construct_critic_observation(obs, self.all_done)

        return obs, critic_obs, reward, terminated, truncated, info

    def _wrap_gym_reset(self,
                        obs,
                        info):
        """
            A method defining how to wrap our enviornment
            reset.

            Arguments:
                obs        The agent observations.
                info       The agent info.

            Returns:
                A tuple of form (obs, critic_obs) s.t.
                each is a dictionary.
        """
        agent_id = self.get_agent_id()

        #
        # HACK: some environments are buggy and don't follow their
        # own rules!
        #
        obs = obs.reshape(self.observation_space[agent_id].shape)
        obs = {agent_id : obs}

        done = {agent_id : False}
        self._update_done_agents(done)

        if self.add_agent_ids:
            obs = self._add_agent_ids_to_obs(obs)

        critic_obs = self._construct_critic_observation(obs, done)

        return obs, critic_obs


class MultiAgentGymWrapper(PPOGymWrapper):
    """
        A wrapper for multi-agent gym environments.

        IMPORTANT: The following assumptions are made about the gym
        environment:

            1. All agent observations, actions, etc. are given in tuples
               s.t. each entry in the tuple corresponds to an agent.
            2. All agents must step at once. If an agent "dies", it still
               will return information every step.
    """

    def __init__(self,
                 env,
                 test_mode     = False,
                 add_agent_ids = True,
                 **kw_args):
        """
            Arguments:
                env            The gym environment to wrap.
                test_mode      Are we in test mode?
                add_agent_ids  Should we add agent ids to the agent
                               observations?
        """
        super(MultiAgentGymWrapper, self).__init__(
            env,
            test_mode,
            add_agent_ids = add_agent_ids,
            **kw_args)

    def _define_agent_ids(self):
        """
            Define our agent_ids.
        """
        self.num_agents = len(self.env.observation_space)
        self.agent_ids  = tuple(f"agent{i}" for i in range(self.num_agents))

    def _define_multi_agent_spaces(self):
        """
            Define our multi-agent spaces.
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

    def _unwrap_action(self,
                       actions):
        """
            An method defining how to unwrap an action.

            Arguments:
                A dictionary mapping agent ids to actions.

            Returns:
                A tuple of actions.
        """
        gym_actions = np.array([None] * self.num_agents)

        for a_idx, a_id in enumerate(self.agent_ids):
            env_action = actions[a_id]
            gym_actions[a_idx] = env_action

        return tuple(gym_actions)

    def _apply_step_restrictions(self, terminated, truncated):
        """
        Apply any desired restrictions to the allowable environment
        steps. This can alter the terminated and truncated dictionaries.

        Parameters:
        -----------
        terminated: dict
            A dictionary mapping agent ids to termination status.
        truncated: dict
            A dictionary mapping agent ids to truncation status.

        Returns:
        --------
        tuple:
            terminated and truncated dictionaries.
        """
        self.step_count += 1

        if self.step_count >= self.max_steps:
            for agent_idx in range(len(terminated)):
                terminated[agent_idx] = True
                truncated[agent_idx]  = False

            self.step_count = 0

        return terminated, truncated

    def _wrap_gym_step(self,
                       obs,
                       reward,
                       terminated,
                       truncated,
                       info):
        """
            A method defining how to wrap our enviornment
            step.

            Arguments:
                obs         The agent observations.
                reward      The agent rewards.
                terminated  The agent termination flags.
                truncated   The agent truncated flags.
                info        The agent info.

            Returns:
                A tuple of form (obs, critic_obs, reward,
                terminated, truncated, info) s.t. each is a dictionary.
        """
        wrapped_obs        = {}
        wrapped_reward     = {}
        wrapped_terminated = {}
        wrapped_truncated  = {}
        wrapped_info       = {}
        done_agents        = {}
        done_array         = np.zeros(len(self.agent_ids)).astype(bool)

        if truncated.any() and not truncated.all():
            msg  = "ERROR: truncation for one but not all agents in an "
            msg += "environment is not currently supported."
            rank_print(msg)
            comm.Abort()

        terminated, truncated = self._apply_step_restrictions(
            terminated, truncated)

        for a_idx, a_id in enumerate(self.agent_ids):
            agent_obs         = obs[a_idx]
            agent_reward      = reward[a_idx]
            agent_terminated  = terminated[a_idx]
            agent_truncated   = truncated[a_idx]
            agent_info        = info
            done_array[a_idx] = truncated[a_idx] or terminated[a_idx]

            #
            # HACK: some environments are buggy and don't follow their
            # own rules!
            #
            agent_obs = agent_obs.reshape(self.enforced_obs_shape[a_id])

            if type(agent_reward) == np.ndarray:
                agent_reward = agent_reward[0]

            agent_reward = np.float32(agent_reward)

            wrapped_obs[a_id]        = agent_obs
            wrapped_reward[a_id]     = agent_reward
            wrapped_info[a_id]       = agent_info
            wrapped_terminated[a_id] = agent_terminated
            wrapped_truncated[a_id]  = agent_truncated
            done_agents[a_id]        = agent_terminated or agent_truncated

        if done_array.all():
            self.all_done = True
        else:
            self.all_done = False

        if self.add_agent_ids:
            wrapped_obs = self._add_agent_ids_to_obs(wrapped_obs)

        self._update_done_agents(done_agents)

        wrapped_obs, wrapped_reward, wrapped_terminated, wrapped_truncated, wrapped_info = \
            self._apply_death_mask(
                wrapped_obs,
                wrapped_reward,
                wrapped_terminated,
                wrapped_truncated,
                wrapped_info)

        critic_obs  = self._construct_critic_observation(
            wrapped_obs, done_agents)

        return (wrapped_obs, critic_obs,
            wrapped_reward, wrapped_terminated,
            wrapped_truncated, wrapped_info)

    def _wrap_gym_reset(self,
                        obs,
                        info):
        """
            A method defining how to wrap our enviornment
            reset.

            Arguments:
                obs        The agent observations.
                info       The agent info.

            Returns:
                A tuple of form (obs, critic_obs) s.t.
                each is a dictionary.
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

        self._update_done_agents(wrapped_done)

        if self.add_agent_ids:
            wrapped_obs = self._add_agent_ids_to_obs(wrapped_obs)

        critic_obs = self._construct_critic_observation(
            wrapped_obs, wrapped_done)

        return wrapped_obs, critic_obs

"""
    Wrappers for Abmarl environments.
    https://abmarl.readthedocs.io/en/latest/index.html
"""
from ppo_and_friends.environments.ppo_env_wrappers import PPOEnvironmentWrapper
from ppo_and_friends.environments.action_wrappers import BoxIntActionEnvironment
from ppo_and_friends.utils.spaces import gym_space_to_gymnasium_space
from abmarl.sim.agent_based_simulation import ActingAgent, Agent, ObservingAgent
from gymnasium.spaces import Dict, Box
import gymnasium as gym
import copy
import time
import numpy as np
import matplotlib.pyplot as plt

class AbmarlWrapper(PPOEnvironmentWrapper, BoxIntActionEnvironment):

    def __init__(self,
                 env,
                 **kw_args):
        """
            Initialize our environment.

            NOTE: All Abmarl environments need to be wrapped in the
            FlattenWrapper, which is provided by Abmarl.

            Arguments:
                env     An instance of the Abmarl environment.
        """
        super().__init__(
            env = env,
            **kw_args)

        self.fig = None
        self.need_action_wrap = False

        #
        # Box int is an odd action space that's very common in Abmarl.
        # What we really want here is a multi discrete action space,
        # so we're going to wrap our actions if needed.
        #
        for agent_id in self.action_space:
            space = self.action_space[agent_id]

            if type(space) == Box and np.issubdtype(space.dtype, np.integer):
                self._wrap_action_space(self.action_space)
                self.need_action_wrap = True
                break

    def _define_agent_ids(self):
        """
            Define our agent ids.
        """
        self.agent_ids = tuple(agent.id for agent in
            self.env.sim.agents.values() if isinstance(agent, Agent))

        self.num_agents = len(self.agent_ids)

    def _define_multi_agent_spaces(self):
        """
            Define our multi-agent observation and action spaces.
        """

        if getattr(self.env, "observation_space", None) is not None:
            #
            # NOTE: I think abmarl does some funny business behind the scenes,
            # and we need to deep copy this space here.
            #
            self.observation_space = copy.deepcopy(self.env.observation_space)
        else:
            self.observation_space = Dict({
                agent.id: gym_space_to_gymnasium_space(agent.observation_space)
                for agent in self.env.sim.agents.values()
                if isinstance(agent, ObservingAgent)
            })

        if self.add_agent_ids:
            for a_id in self.observation_space:
                self.observation_space[a_id] = \
                    self._expand_space_for_ids(
                        self.observation_space[a_id])

        if getattr(self.env, "action_space", None) is not None:
            self.action_space = self.env.action_space
        else:
            self.action_space = Dict({
                agent.id: gym_space_to_gymnasium_space(agent.action_space)
                for agent in self.env.sim.agents.values()
                if isinstance(agent, ActingAgent)
            })

        obs_keys    = set(self.observation_space.keys())
        action_keys = set(self.action_space.keys())
        all_keys    = obs_keys.union(action_keys)

        if len(all_keys) != len(obs_keys):
            msg  = "ERROR: having observation and action spaces "
            msg += "with different agent ids is currently unsupported. "
            msg += f"Observation keys: {obs_keys}"
            msg += f"Action keys: {action_keys}"
            rank_print(msg)
            comm.Abort()

        #
        # FIXME: Temporary hack! We need all spaces to be from the
        # gymnasium module, but Abmarl currently is using gym.
        #
        self.action_space = gym_space_to_gymnasium_space(self.action_space)
        self.observation_space = gym_space_to_gymnasium_space(self.observation_space)

    def _get_all_done(self, done):
        """
            Determine whether or not all agents are done.

            Arguments:
                done    A dictionary mapping agent ids to bools.

            Returns:
                True iff all agents are done.
        """
        for agent_id in done:
            if not done[agent_id]:
                return False
        return True

    def step(self, actions):
        """
            Take a step in the simulation.

            Arguments:
                actions    A dictionary mapping agent ids to actions.

            Returns:
                A tuple of form (actor_observation, critic_observation,
                reward, done, info)
        """
        # TODO: need to implement turn masking.
        actions = self._filter_done_agent_actions(actions)

        if self.need_action_wrap:
            obs, reward, terminated, info = self._action_wrapped_step(actions)
        else:
            obs, reward, terminated, info = self.env.step(actions)

        truncated = {}
        for key in terminated:
            truncated[key] = False

        self.all_done = self._get_all_done(terminated)
        self._update_done_agents(terminated)

        if self.add_agent_ids:
            obs = self._add_agent_ids_to_obs(obs)

        obs, reward, terminated, truncated, info = self._apply_death_mask(
            obs, reward, terminated, truncated, info)

        critic_obs = self._construct_critic_observation(
            obs, terminated)

        return obs, critic_obs, reward, terminated, truncated, info

    def reset(self):
        """
            Reset the environment.

            Returns:
                A tuple of form (actor_observation, critic_observation).
        """
        obs = self.env.reset()

        if self.add_agent_ids:
            obs = self._add_agent_ids_to_obs(obs)

        done = {a_id : False for a_id in obs}
        self._update_done_agents(done)

        critic_obs = self._construct_critic_observation(
            obs, done)

        return obs, critic_obs

    def render(self, frame_pause = 0.0, **kw_args):
        """
            Render the environment.

            Arguments:
                frame_pause    Sleep frame_pause seconds before rendering.
        """

        if frame_pause > 0.0:
            time.sleep(frame_pause)

        if self.fig is None:
            self.fig, _ = plt.subplots()

        self.env.render(fig=self.fig)
        plt.pause(1e-12)

    def seed(self, seed):
        """
            Set the Abmarl seed. It looks like the Abmarl tests just
            use numpy to set the seed, so that's what we'll do.

            Arguments:
                seed (int)    The seed to set.
        """
        np.random.seed(seed)

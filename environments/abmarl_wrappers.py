from ppo_and_friends.environments.ppo_env_wrappers import PPOEnvironmentWrapper
from ppo_and_friends.environments.action_wrappers import BoxIntActionEnvironment
from abmarl.sim.agent_based_simulation import ActingAgent, Agent, ObservingAgent
from gym.spaces import Dict, Box
import numpy as np
import matplotlib.pyplot as plt

class AbmarlWrapper(PPOEnvironmentWrapper, BoxIntActionEnvironment):

    def __init__(self,
                 env,
                 test_mode         = False,
                 add_agent_ids     = False,
                 critic_view       = "policy",
                 policy_mapping_fn = None,
                 **kw_args):
        """
        """
        self.fig, _ = plt.subplots()

        self.agent_ids = tuple(agent.id for agent in env.sim.agents.values() if
            isinstance(agent, Agent))

        super(AbmarlWrapper, self).__init__(
            env               = env,
            test_mode         = test_mode,
            add_agent_ids     = add_agent_ids,
            critic_view       = critic_view,
            policy_mapping_fn = policy_mapping_fn,
            *kw_args)

        self.need_action_wrap = False

        for agent_id in self.action_space:
            space = self.action_space[agent_id]

            if type(space) == Box and np.issubdtype(space.dtype, np.integer):
                self._wrap_action_space(self.action_space)
                self.need_action_wrap = True
                break

    def _define_multi_agent_spaces(self):
        """
        """
        if getattr(self.env, "observation_space", None) is not None:
            self.observation_space = self.env.observation_space
        else:
            self.observation_space = Dict({
                agent.id: agent.observation_space
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
                agent.id: agent.action_space
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

    def _get_all_done(self, done):
        """
        """
        for agent_id in done:
            if not done[agent_id]:
                return False
        return True

    def step(self, actions):
        """
        """
        # FIXME: need to add agents that are missing (death or turns)

        if self.need_action_wrap:
            obs, reward, done, info = self._action_wrapped_step(actions)
        else:
            obs, reward, done, info = self.env.step(actions)

        self.all_done = self._get_all_done(done)

        if self.add_agent_ids:
            obs = self._add_agent_ids_to_obs(obs)

        obs        = self._apply_death_mask(obs, done)
        critic_obs = self._construct_critic_observation(
            obs, done)

        return obs, critic_obs, reward, done, info

    def reset(self):
        """
        """
        obs = self.env.reset()

        if self.add_agent_ids:
            obs = self._add_agent_ids_to_obs(obs)

        done = {a_id : False for a_id in obs}

        critic_obs = self._construct_critic_observation(
            obs, done)

        return obs, critic_obs

    def render(self, **kw_args):
        """
        """
        #FIXME: there are still some issues with this.
        self.env.render(fig=self.fig)


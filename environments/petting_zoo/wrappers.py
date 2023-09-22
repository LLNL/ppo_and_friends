"""
    This module contains wrappers for petting zoo environments.
"""
from ppo_and_friends.environments.ppo_env_wrappers import PPOEnvironmentWrapper
from ppo_and_friends.utils.mpi_utils import rank_print

from mpi4py import MPI
comm      = MPI.COMM_WORLD
rank      = comm.Get_rank()
num_procs = comm.Get_size()


class ParallelZooWrapper(PPOEnvironmentWrapper):
    """
        A wrapper for parallel petting zoo environments.
    """
    def __init__(self, *args, **kw_args):

        super(ParallelZooWrapper, self).__init__(
            *args, **kw_args)

        self.random_seed = None

    def _define_agent_ids(self):
        """
            Define our agent_ids.
        """
        self.num_agents = len(self.env.possible_agents)
        self.agent_ids  = tuple(self.env.possible_agents)

    def _define_multi_agent_spaces(self):
        """
            Define our multi-agent spaces.
        """
        for a_id in self.agent_ids:
            if self.add_agent_ids:
                self.observation_space[a_id] = \
                    self._expand_space_for_ids(self.env.observation_space(a_id))
            else:
                self.observation_space[a_id] = self.env.observation_space(a_id)

            self.action_space[a_id] = self.env.action_space(a_id)

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

    def _conform_actions(self, actions):
        """
            Make sure that our actions conform to the environment's
            requirements.

            Arguments:
                actions (dict)    The dictionary mapping agents to actions.

            Returns:
                A copy of actions where the actions are reshaped to match
                the agent's action space.
        """
        c_actions = {}
        for agent_id in actions:
            c_actions[agent_id] = \
                actions[agent_id].reshape(self.action_space[agent_id].shape)
        return c_actions

    def step(self, actions):
        """
            Take a step in the simulation.

            Arguments:
                actions    A dictionary mapping agent ids to actions.

            Returns:
                A tuple of form (actor_observation, critic_observation,
                reward, terminal, truncated, info)
        """
        actions = self._filter_done_agent_actions(actions)

        obs, reward, terminal, truncated, info = \
            self.env.step(self._conform_actions(actions))

        done = self._get_done_dict(terminal, truncated)

        self.all_done = self._get_all_done(done)
        self._update_done_agents(done)

        if self.add_agent_ids:
            obs = self._add_agent_ids_to_obs(obs)

        obs, reward, terminal, truncated, info = self._apply_death_mask(
            obs, reward, terminal, truncated, info)

        critic_obs = self._construct_critic_observation(
            obs, self.agents_done)

        return obs, critic_obs, reward, terminal, truncated, info

    def reset(self):
        """
            Reset the environment.

            Returns:
                A tuple of form (actor_observation, critic_observation).
        """
        obs , info = self.env.reset(seed = self.random_seed)

        #
        # Zoo requires the random seed to be set
        # when calling reset. Since we don't want the same exact
        # episode to reply every time we reset, we increment the
        # seed. This retains reproducibility while allow each episode
        # to vary.
        #
        if self.random_seed != None:
            self.random_seed += 1

        if self.add_agent_ids:
            obs = self._add_agent_ids_to_obs(obs)

        self._reset_done_agents()

        done = {a_id : False for a_id in obs}
        critic_obs = self._construct_critic_observation(
            obs, done)

        return obs, critic_obs

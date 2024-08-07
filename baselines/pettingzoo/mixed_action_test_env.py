"""
A very simple PettingZoo environment for testing mixed action spaces.
"""
import numpy as np
import functools
import torch
import random
import gymnasium as gym
import gymnasium.spaces as spaces
from gymnasium.utils import seeding
from pettingzoo import ParallelEnv
from pettingzoo.utils import parallel_to_aec, wrappers
from ppo_and_friends.environments.petting_zoo.wrappers import ParallelZooWrapper
from ppo_and_friends.runners.env_runner import GymRunner
from ppo_and_friends.networks.ppo_networks.feed_forward import FeedForwardNetwork
from ppo_and_friends.utils.schedulers import *
from ppo_and_friends.utils.spaces import FlatteningTuple
from ppo_and_friends.utils.misc import get_flattened_space_length, get_space_dtype_str
import torch.nn as nn
from ppo_and_friends.runners.runner_tags import ppoaf_runner
from gymnasium.spaces import Discrete, MultiDiscrete


def env(render_mode=None):
    """
    The env function often wraps the environment in wrappers by default.
    You can find full documentation for these methods
    elsewhere in the developer documentation.
    """
    internal_render_mode = render_mode if render_mode != "ansi" else "human"
    env = raw_env(render_mode=internal_render_mode)

    if render_mode == "ansi":
        env = wrappers.CaptureStdoutWrapper(env)

    # this wrapper helps error handling for discrete action spaces
    env = wrappers.AssertOutOfBoundsWrapper(env)

    # Provides a wide vareity of helpful user errors
    # Strongly recommended
    env = wrappers.OrderEnforcingWrapper(env)

    return env


def raw_env(render_mode=None):
    """
    To support the AEC API, the raw_env() function just uses the from_parallel
    function to convert from a ParallelEnv to an AEC env
    """
    env = parallel_env(render_mode=render_mode)
    env = parallel_to_aec(env)
    return env

class MixedActionMirrorAgent():

    def __init__(self, action_space, observation_space, seed):
        assert issubclass(type(action_space), FlatteningTuple)

        self.action_space      = action_space
        self.observation_space = observation_space

        self.seed(seed)

        #
        # We will reward agents for aligning their actions with
        # these "target actions". It's just a way to make sure
        # the mixed action space is functional.
        #
        self.target = self.action_space.sample()

    def seed(self, seed):
        self.action_space.seed(seed)

class MixedActionMirror(ParallelEnv):
    """
    A simple game to help debug and test mixed-action space functionality.
    In this game, agents are rewarded for their actions mirroring a "target"
    action that is a sample of their action space.
    """

    def __init__(self, max_steps=256, num_agents=10, omit_spaces=[]):

        for i in range(len(omit_spaces)):
            omit_spaces[i] = int(omit_spaces[i])

        self.max_steps    = max_steps
        self.current_step = 0

        avail_spaces = [\
            spaces.Box(-1, 1, (2,)),
            spaces.Discrete(3),
            spaces.Box(-10, 2, (3,)),
            spaces.MultiBinary(5),
            spaces.MultiDiscrete([5, 10, 20]),
        ]

        mixed_spaces = []

        for i in range(len(avail_spaces)):
            if i not in omit_spaces:
                mixed_spaces.append(avail_spaces[i])

        action_space = FlatteningTuple(mixed_spaces)
        obs_size     = get_flattened_space_length(action_space)

        low  = np.inf
        high = -np.inf
        for space in mixed_spaces:
            space_dtype = get_space_dtype_str(space)

            if space_dtype == "discrete":
                low  = min(0, low)
                high = max(space.n - 1, high)

            elif space_dtype == "multi-discrete":
                low  = min(0, low)
                high = max(space.nvec.max() - 1, high)

            elif space_dtype == "multi-binary":
                low  = min(0, low)
                high = max(1, high)

            elif space_dtype == "continuous":
                low  = min(space.low.min(), low)
                high = max(space.high.max(), high)

        observation_space = spaces.Box(low, high, (obs_size,))

        self.agents = {}
        for i in range(num_agents):
            self.agents[f"agent_{i}"] = \
                MixedActionMirrorAgent(action_space, observation_space, i)

        self.possible_agents = [a_id for a_id in self.agents]

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent_id):
        return self.agents[agent_id].observation_space

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent_id):
        return self.agents[agent_id].action_space

    def seed(self, seed=None):
        if seed is not None and seed >= 0:
            np.random.seed(seed)
            random.seed(seed)
      
        for agent_id in self.agents:
            self.agents[agent_id].seed(seed)

        return [seed]

    def step(self, actions):
        obs        = {}
        reward     = {}
        terminated = {}
        truncated  = {}
        info       = {}

        #
        # In this simple game, we reward agents using actions that
        # mirror their "target", which is just a sample of their action space.
        #
        for agent_id in actions:
            obs[agent_id]    = self.agents[agent_id].target - actions[agent_id]
            reward[agent_id] = -np.square(actions[agent_id] - self.agents[agent_id].target).mean()

            if self.current_step >= self.max_steps:
                terminated[agent_id] = True
            else:
                terminated[agent_id] = False

            truncated[agent_id] = False
            info[agent_id]      = {}

        self.current_step += 1

        return obs, reward, terminated, truncated, info

    def reset(self, seed=None, options={}, *args, **kw_args):
        self.seed(seed)
        self.current_step = 0

        obs  = {}
        info = {}

        for agent_id in self.agents:
            obs[agent_id] = np.ones(self.agents[agent_id].target.size)

        return obs, info

    def render(self, mode = 'human'):
        return

@ppoaf_runner
class MixedActionMirrorRunner(GymRunner):

    def add_cli_args(self, parser):
        """
        Define extra args that will be added to the ppoaf command.

        Parameters:
        -----------
        parser: argparse.ArgumentParser
            The parser from ppoaf.

        Returns:
        --------
        argparse.ArgumentParser:
            The same parser as the input with potentially new arguments added.
        """
        parser.add_argument("--learning_rate", type=float, default=0.0005)
        parser.add_argument("--omit_spaces", nargs="+", default=[])
        return parser

    def run(self):

        policy_map = lambda x : 'agent'

        env_generator = lambda : \
            ParallelZooWrapper(
                MixedActionMirror(omit_spaces = self.cli_args.omit_spaces),
                add_agent_ids     = True,
                critic_view       = "local",
                policy_mapping_fn = policy_map)

        actor_kw_args = {}

        actor_kw_args["activation"]  = nn.LeakyReLU()
        actor_kw_args["hidden_size"] = 64

        critic_kw_args = actor_kw_args.copy()
        critic_kw_args["hidden_size"] = 64

        policy_args = {\
            "ac_network"       : FeedForwardNetwork,
            "actor_kw_args"    : actor_kw_args,
            "critic_kw_args"   : critic_kw_args,
            "lr"               : self.cli_args.learning_rate,
        }

        policy_settings = { 
            "agent" : \
                (None,
                 env_generator().observation_space["agent_0"],
                 env_generator().critic_observation_space["agent_0"],
                 env_generator().action_space["agent_0"],
                 policy_args),
        }

        self.run_ppo(env_generator       = env_generator,
                     policy_settings     = policy_settings,
                     policy_mapping_fn   = policy_map,
                     max_ts_per_ep       = 32,
                     epochs_per_iter     = 15,
                     ts_per_rollout      = 256,
                     batch_size          = 256,
                     normalize_obs       = False,
                     obs_clip            = None,
                     normalize_rewards   = False,
                     reward_clip         = None,
                     **self.kw_run_args)

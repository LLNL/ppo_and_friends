import gymnasium as gym
from ppo_and_friends.environments.gym.wrappers import SingleAgentGymWrapper
from ppo_and_friends.policies.utils import get_single_policy_defaults
from ppo_and_friends.runners.env_runner import GymRunner
from ppo_and_friends.networks.ppo_networks.feed_forward import FeedForwardNetwork
from ppo_and_friends.utils.schedulers import *
import torch.nn as nn
from ppo_and_friends.runners.runner_tags import ppoaf_runner

@ppoaf_runner
class HumanoidStandUpRunner(GymRunner):

    def run(self):
        #
        # NOTE: this is an UNSOVLED environment.
        #
        env_generator = lambda : \
            SingleAgentGymWrapper(gym.make('HumanoidStandup-v4',
                render_mode = self.get_gym_render_mode()))

        #
        #    Positions: 22
        #    Velocities: 23
        #    Center of mass based on inertia (?): 140
        #    Center of mass based on velocity (?): 84
        #    Actuator forces (?): 23
        #    Contact forces: 84
        #
        # UPDATE: more complete information on the observations cane be found
        # here:
        # https://github.com/openai/gym/blob/master/gym/envs/mujoco/humanoidstandup.py
        #
        actor_kw_args = {}
        actor_kw_args["activation"]       = nn.Tanh()
        actor_kw_args["distribution_min"] = -0.4
        actor_kw_args["distribution_max"] = 0.4
        actor_kw_args["hidden_size"]      = 256

        critic_kw_args = actor_kw_args.copy()
        critic_kw_args["hidden_size"] = 512

        lr = LinearScheduler(
            status_key    = "iteration",
            status_max    = 200,
            max_value     = 0.0003,
            min_value     = 0.0001)

        policy_args = {\
            "ac_network"       : FeedForwardNetwork,
            "actor_kw_args"    : actor_kw_args,
            "critic_kw_args"   : critic_kw_args,
            "lr"               : lr,
            "target_kl"        : 0.015,
        }

        policy_settings, policy_mapping_fn = get_single_policy_defaults(
            env_generator = env_generator,
            policy_args   = policy_args)

        self.run_ppo(env_generator      = env_generator,
                     policy_settings    = policy_settings,
                     policy_mapping_fn  = policy_mapping_fn,
                     batch_size         = 512,
                     max_ts_per_ep      = 32,
                     epochs_per_iter    = 8,
                     ts_per_rollout     = 512,
                     obs_clip           = None,
                     reward_clip        = (-10., 10.),
                     **self.kw_run_args)

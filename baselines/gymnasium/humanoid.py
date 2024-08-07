import gymnasium as gym
from ppo_and_friends.environments.gym.wrappers import SingleAgentGymWrapper
from ppo_and_friends.policies.utils import get_single_policy_defaults
from ppo_and_friends.runners.env_runner import GymRunner
from ppo_and_friends.networks.ppo_networks.feed_forward import FeedForwardNetwork
from ppo_and_friends.utils.schedulers import *
import torch.nn as nn
from ppo_and_friends.runners.runner_tags import ppoaf_runner

@ppoaf_runner
class HumanoidRunner(GymRunner):

    def run(self):
        env_generator = lambda : \
            SingleAgentGymWrapper(gym.make('Humanoid-v4',
                render_mode = self.get_gym_render_mode()))

        #
        # Humanoid observations are a bit mysterious. See
        # https://github.com/openai/gym/issues/585
        # Here's a best guess:
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

        # TODO: the current settings work pretty well, but it
        # takes a while to train. Can we do better? Some things
        # that need more exploring:
        #    std offset: is the default optimal?
        #    activation: How does leaky relu do?
        #    target_kl: we could experiment more with this.
        #    obs_clip: this seems to negatively impact results. Does that hold?
        #    entropy: we could allow entropy reg, but I'm guessing it won't help
        #             too much.
        #
        actor_kw_args["activation"]       = nn.Tanh()
        actor_kw_args["distribution_min"] = -0.4
        actor_kw_args["distribution_max"] = 0.4
        actor_kw_args["hidden_size"]      = 256

        critic_kw_args = actor_kw_args.copy()
        critic_kw_args["hidden_size"] = 512

        lr = 0.0001

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
                     epochs_per_iter    = 8,
                     max_ts_per_ep      = 16,
                     ts_per_rollout     = 512,
                     reward_clip        = (-10., 10.),
                     **self.kw_run_args)

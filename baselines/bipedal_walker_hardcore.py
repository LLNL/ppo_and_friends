import gymnasium as gym
from ppo_and_friends.environments.gym.wrappers import SingleAgentGymWrapper
from ppo_and_friends.policies.utils import get_single_policy_defaults
from ppo_and_friends.runners.env_runner import GymRunner
from ppo_and_friends.networks.ppo_networks.feed_forward import FeedForwardNetwork
from ppo_and_friends.utils.schedulers import *
import torch.nn as nn
from ppo_and_friends.runners.runner_tags import ppoaf_runner

@ppoaf_runner
class BipedalWalkerHardcoreRunner(GymRunner):

    def run(self):
        env_generator = lambda : \
            SingleAgentGymWrapper(gym.make('BipedalWalkerHardcore-v3',
                render_mode = self.get_gym_render_mode()))

        actor_kw_args = {}
        actor_kw_args["std_offset"]  = 0.1
        actor_kw_args["activation"]  = nn.LeakyReLU()
        actor_kw_args["hidden_size"] = 256

        critic_kw_args = actor_kw_args.copy()
        critic_kw_args["hidden_size"] = 512

        #
        # This environment is a pretty challenging one and can be
        # very finicky. Learning rate and reward clipping have a
        # pretty powerfull impact on results, and it can be tricky
        # to get these right. Here's what I've found works best:
        #
        #   1. First, run the problem with a pretty standard learning
        #      rate (0.0001 works well), and use a conservative reward
        #      clipping of (-1, 10). Clipping the reward at -1 results
        #      in the agent learning a good gait pretty early on.
        #
        #   2. After a while (roughly 7000->8000 iterations), the agent should
        #      have a pretty solid policy. Running tests will show that
        #      it can regularly reach scores over 300, but averaging
        #      over 100 runs will likely be in the 200s. My impression
        #      here is that the -1 reward clipping, while allowing the
        #      agent to learn a good gait quickly, also causes the agent
        #      be less concerned with failures. So, at this point, I find
        #      that adjusting the lower bound of the clip to the standard
        #      -10 value allows the agent to learn that falling is
        #      actually really bad. I also lower the learning rate here
        #      to help with stability in this last phase. This last bit
        #      of learning can take a while (~10,000 -> 11,000 iterations).
        #
        # The above is all automated with the settings used below. I typically
        # run with 4 processors. The resulting policy can regularly reach average
        # scores of 320+ over 100 test runs.
        #
        lr = LinearStepScheduler(
            status_key      = "iteration",
            initial_value   = 0.0001,
            status_triggers = [3900,],
            step_values     = [0.00001,])

        reward_clip_min = LinearStepScheduler(
            status_key      = "iteration",
            initial_value   = -1.,
            status_triggers = [4000,],
            step_values     = [-10.,])

        bs_clip_min = LinearStepScheduler(
            status_key      = "iteration",
            initial_value   = -1.,
            status_triggers = [4000,],
            step_values     = [-10.,])

        policy_args = {\
            "ac_network"       : FeedForwardNetwork,
            "actor_kw_args"    : actor_kw_args,
            "critic_kw_args"   : critic_kw_args,
            "lr"               : lr,
            "bootstrap_clip"   : (bs_clip_min, 10.),
        }

        policy_settings, policy_mapping_fn = get_single_policy_defaults(
            env_generator = env_generator,
            policy_args   = policy_args)

        ts_per_rollout = self.get_adjusted_ts_per_rollout(512)

        self.run_ppo(env_generator      = env_generator,
                     policy_settings    = policy_settings,
                     policy_mapping_fn  = policy_mapping_fn,
                     batch_size         = 512,
                     max_ts_per_ep      = 32,
                     ts_per_rollout     = ts_per_rollout,
                     normalize_obs      = True,
                     normalize_rewards  = True,
                     obs_clip           = (-10., 10.),
                     reward_clip        = (reward_clip_min, 10.),
                     **self.kw_run_args)

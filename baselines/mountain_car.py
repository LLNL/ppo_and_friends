import gymnasium as gym
from ppo_and_friends.environments.gym.wrappers import SingleAgentGymWrapper
from ppo_and_friends.policies.utils import get_single_policy_defaults
from ppo_and_friends.runners.env_runner import GymRunner
from ppo_and_friends.networks.ppo_networks.feed_forward import FeedForwardNetwork
from ppo_and_friends.utils.schedulers import *
import torch.nn as nn
from ppo_and_friends.runners.runner_tags import ppoaf_runner

@ppoaf_runner
class MountainCarRunner(GymRunner):

    def run(self):

        env_generator = lambda : \
            SingleAgentGymWrapper(gym.make('MountainCar-v0',
                render_mode = self.get_gym_render_mode()))

        actor_kw_args = {"activation" : nn.LeakyReLU()}
        actor_kw_args["hidden_size"]  = 32
        actor_kw_args["hidden_depth"] = 2

        critic_kw_args = actor_kw_args.copy()
        critic_kw_args["hidden_size"]  = 64
        critic_kw_args["hidden_depth"] = 2

        icm_kw_args = {}
        icm_kw_args["encoded_obs_dim"] = 0
        icm_kw_args["inverse_hidden_depth"] = 2
        icm_kw_args["inverse_hidden_size"]  = 32
        icm_kw_args["forward_hidden_depth"] = 2
        icm_kw_args["forward_hidden_size"]  = 32

        lr = 0.0003

        policy_args = {\
            "ac_network"       : FeedForwardNetwork,
            "actor_kw_args"    : actor_kw_args,
            "critic_kw_args"   : critic_kw_args,
            "lr"               : lr,

            #
            # NOTE: I find that that the chosen bootstrap clip
            # is VERY important in this environment.
            #
            "bootstrap_clip"   : (-.01, 0.0),
            "enable_icm"       : True,
            "icm_kw_args"      : icm_kw_args,
            "icm_lr"           : 0.0003,
        }

        policy_settings, policy_mapping_fn = get_single_policy_defaults(
            env_generator = env_generator,
            policy_args   = policy_args)

        save_when = ChangeInStateScheduler(
            status_key     = "extrinsic score avg",
            status_preface = "single_agent",
            compare_fn     = np.greater_equal,
            persistent     = True)

        ts_per_rollout = self.get_adjusted_ts_per_rollout(200)

        #
        # NOTE: This environment performs dramatically  better when
        # max_ts_per_ep is set to the total timesteps allowed by the
        # environment. It's not 100% clear to me why this is the case.
        # We should probably explore this a bit more. MountainCarContinuous
        # doesn't seem to exhibit this behavior, so it's unlikely an issue
        # with ICM.
        # Also, the extrinsic reward weight fraction is very important
        # for good performance.
        #
        self.run_ppo(env_generator      = env_generator,
                     save_when          = save_when,
                     policy_settings    = policy_settings,
                     policy_mapping_fn  = policy_mapping_fn,
                     ts_per_rollout     = ts_per_rollout,
                     epochs_per_iter    = 32,
                     max_ts_per_ep      = 200,
                     ext_reward_weight  = 1./100.,
                     normalize_obs      = False,
                     normalize_rewards  = False,
                     normalize_values   = False,
                     obs_clip           = None,
                     reward_clip        = None,
                     **self.kw_run_args)

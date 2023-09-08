from ppo_and_friends.environments.abmarl.wrappers import AbmarlWrapper
from ppo_and_friends.policies.utils import get_single_policy_defaults
from ppo_and_friends.runners.env_runner import EnvironmentRunner
from ppo_and_friends.networks.ppo_networks.feed_forward import FeedForwardNetwork
from ppo_and_friends.utils.schedulers import *
from ppo_and_friends.environments.abmarl.envs.reach_the_target import abmarl_rtt_env
import torch.nn as nn
from ppo_and_friends.runners.runner_tags import ppoaf_runner

@ppoaf_runner
class AbmarlReachTheTargetRunner(EnvironmentRunner):

    def run(self):
        actor_kw_args = {}
        actor_kw_args["activation"]  = nn.LeakyReLU()
        actor_kw_args["hidden_size"] = 64

        critic_kw_args = actor_kw_args.copy()
        critic_kw_args["hidden_size"] = 128

        lr = LinearScheduler(
            status_key     = "iteration",
            status_max     = 100,
            max_value      = 0.0003,
            min_value      = 0.0001)

        policy_args = {\
            "ac_network"         : FeedForwardNetwork,
            "actor_kw_args"      : actor_kw_args,
            "critic_kw_args"     : critic_kw_args,
            "lr"                 : lr,
            "bootstrap_clip"     : (-10, 10),
        }

        def policy_mapping_fn(agent_id):
            return 'runner' if agent_id.startswith('runner') else 'target'

        env_generator = lambda : \
                AbmarlWrapper(env               = abmarl_rtt_env,
                              policy_mapping_fn = policy_mapping_fn,
                              critic_view       = "policy")

        #
        # This environment is multi-agent and requires different policies.
        #
        policy_settings = {
            "target" :
            (None,
             env_generator().observation_space["target"],
             env_generator().critic_observation_space["target"],
             env_generator().action_space["target"],
             policy_args),

            "runner" :
            (None,
             env_generator().observation_space["runner0"],
             env_generator().critic_observation_space["runner0"],
             env_generator().action_space["runner0"],
             policy_args)
        }

        ts_per_rollout = self.get_adjusted_ts_per_rollout(256)

        self.run_ppo(env_generator      = env_generator,
                     policy_settings    = policy_settings,
                     policy_mapping_fn  = policy_mapping_fn,
                     batch_size         = 256,
                     epochs_per_iter    = 20,
                     max_ts_per_ep      = 32,
                     ts_per_rollout     = ts_per_rollout,
                     **self.kw_run_args)

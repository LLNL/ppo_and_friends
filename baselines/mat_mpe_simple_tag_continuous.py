from pettingzoo.mpe import simple_tag_v3
from ppo_and_friends.policies.utils import get_single_policy_defaults
from ppo_and_friends.environments.petting_zoo.wrappers import ParallelZooWrapper
from ppo_and_friends.runners.env_runner import GymRunner
from ppo_and_friends.networks.ppo_networks.feed_forward import FeedForwardNetwork
from ppo_and_friends.utils.schedulers import *
from ppo_and_friends.policies.mat_policy import MATPolicy
import torch.nn as nn
from ppo_and_friends.runners.runner_tags import ppoaf_runner

@ppoaf_runner
class MATMPESimpleTagContinuousRunner(GymRunner):

    def run(self):

        policy_map = lambda name : 'adversary' if 'adversary' in name \
            else 'agent'

        #FIXME: continuous version isn't working
        env_generator = lambda : \
            ParallelZooWrapper(
                simple_tag_v3.parallel_env(
                    num_good=1,
                    num_adversaries=3,
                    num_obstacles=2,
                    max_cycles=128,
                    continuous_actions=True,
                    render_mode=self.get_gym_render_mode()),

                critic_view       = "local",
                policy_mapping_fn = policy_map)

        actor_kw_args = {}

        adversary_kw_args  = {}
        adversary_kw_args["distribution_min"] = 0.0
        adversary_kw_args["distribution_max"] = 1.0

        lr = 0.0003

        ts_per_rollout = self.get_adjusted_ts_per_rollout(256)

        adversary_policy_args = {\
            "mat_kw_args"      : adversary_kw_args,
            "lr"               : lr,
        }

        agent_actor_kw_args  = {}
        agent_actor_kw_args["distribution_min"] = 0.0
        agent_actor_kw_args["distribution_max"] = 1.0
        agent_critic_kw_args = agent_actor_kw_args.copy()
        agent_actor_kw_args["activation"]  = nn.LeakyReLU()
        agent_actor_kw_args["hidden_size"] = 256

        agent_critic_kw_args = agent_actor_kw_args.copy()
        agent_critic_kw_args["hidden_size"] = 512

        agent_policy_args = {\
            "ac_network"       : FeedForwardNetwork,
            "actor_kw_args"    : agent_actor_kw_args,
            "critic_kw_args"   : agent_critic_kw_args,
            "lr"               : lr,
        }

        policy_settings = { 
            "agent" : \
                (None,
                 env_generator().observation_space["agent_0"],
                 env_generator().critic_observation_space["agent_0"],
                 env_generator().action_space["agent_0"],
                 agent_policy_args),
            "adversary" : \
                (MATPolicy,
                 env_generator().observation_space["adversary_0"],
                 env_generator().critic_observation_space["adversary_0"],
                 env_generator().action_space["adversary_0"],
                 adversary_policy_args),
        }

        self.run_ppo(env_generator       = env_generator,
                     policy_settings     = policy_settings,
                     policy_mapping_fn   = policy_map,
                     max_ts_per_ep       = 64,
                     ts_per_rollout      = ts_per_rollout,
                     batch_size          = 256,
                     normalize_obs       = False,
                     obs_clip            = None,
                     normalize_rewards   = False,
                     reward_clip         = None,
                     **self.kw_run_args)

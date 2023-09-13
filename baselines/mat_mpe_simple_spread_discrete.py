from pettingzoo.mpe import simple_spread_v3
from ppo_and_friends.policies.utils import get_single_policy_defaults
from ppo_and_friends.environments.petting_zoo.wrappers import ParallelZooWrapper
from ppo_and_friends.runners.env_runner import GymRunner
from ppo_and_friends.networks.ppo_networks.feed_forward import FeedForwardNetwork
from ppo_and_friends.utils.schedulers import *
import torch.nn as nn
from ppo_and_friends.runners.runner_tags import ppoaf_runner
from ppo_and_friends.policies.mat_policy import MATPolicy

@ppoaf_runner
class MATMPESimpleSpreadDiscreteRunner(GymRunner):

    def run(self):

        policy_map = lambda x : 'agent'

        env_generator = lambda : \
            ParallelZooWrapper(
                simple_spread_v3.parallel_env(
                    N=3,
                    local_ratio=0.5,
                    max_cycles=64,
                    continuous_actions=False,
                    render_mode=self.get_gym_render_mode()),

                add_agent_ids     = False,
                critic_view       = "local",
                policy_mapping_fn = policy_map)

        mat_kw_args = {}
        mat_kw_args["embedding size"] = 64

        lr = 0.0003

        ts_per_rollout = self.get_adjusted_ts_per_rollout(256)

        policy_args = {\
            "mat_kw_args"    : mat_kw_args,
            "lr"             : lr,
        }

        policy_settings = { 
            "agent" : \
                (MATPolicy,
                 env_generator().observation_space["agent_0"],
                 env_generator().critic_observation_space["agent_0"],
                 env_generator().action_space["agent_0"],
                 policy_args),
        }

        save_when = ChangeInStateScheduler(
            status_key     = "extrinsic reward avg",
            status_preface = "agent",
            compare_fn     = np.greater_equal,
            persistent     = True)

        self.run_ppo(env_generator       = env_generator,
                     save_when           = save_when,
                     policy_settings     = policy_settings,
                     policy_mapping_fn   = policy_map,
                     max_ts_per_ep       = 32,
                     epochs_per_iter     = 10,
                     ts_per_rollout      = ts_per_rollout,
                     batch_size          = 256,
                     normalize_obs       = False,
                     obs_clip            = None,
                     normalize_rewards   = False,
                     reward_clip         = None,
                     **self.kw_run_args)

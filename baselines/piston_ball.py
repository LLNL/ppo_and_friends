from pettingzoo.butterfly import pistonball_v6
from ppo_and_friends.policies.utils import get_single_policy_defaults
from ppo_and_friends.environments.petting_zoo.wrappers import PPOParallelZooWrapper
from ppo_and_friends.runners.env_runner import GymRunner
from ppo_and_friends.networks.actor_critic_networks import ZooPixelNetwork
from ppo_and_friends.utils.schedulers import *
import torch.nn as nn

class PistonBallRunner(GymRunner):

    def run(self):
        env_generator = lambda : \
            PPOParallelZooWrapper(
                pistonball_v6.parallel_env(
                    render_mode = self.get_gym_render_mode()),
                #
                # Each agent views the entire screen, so the "local"
                # view is actually global.
                #
                critic_view       = "local",
                policy_mapping_fn = lambda *args : "piston")

        #
        # Extra args for the actor critic models.
        # I find that leaky relu does much better with the lunar
        # lander env.
        #
        actor_kw_args = {}

        actor_kw_args["activation"]  = nn.LeakyReLU()
        critic_kw_args = actor_kw_args.copy()

        lr = 0.0003

        #
        # Running with 2 processors works well here.
        #
        ts_per_rollout = self.get_adjusted_ts_per_rollout(256)

        policy_args = {\
            "ac_network"       : ZooPixelNetwork,
            "actor_kw_args"    : actor_kw_args,
            "critic_kw_args"   : critic_kw_args,
            "lr"               : lr,
        }

        def policy_mapping_fn(agent_id):
            return 'piston'

        policy_settings = { "piston" : \
            (None,
             env_generator().observation_space["piston_0"],
             env_generator().critic_observation_space["piston_0"],
             env_generator().action_space["piston_0"],
             policy_args),
        }

        save_when = ChangeInStateScheduler(
            status_key     = "extrinsic score avg",
            status_preface = "piston",
            compare_fn     = np.greater_equal,
            persistent     = True)

        self.run_ppo(env_generator       = env_generator,
                     save_when           = save_when,
                     policy_settings     = policy_settings,
                     policy_mapping_fn   = policy_mapping_fn,
                     max_ts_per_ep       = 128,
                     ts_per_rollout      = ts_per_rollout,
                     batch_size          = 256,
                     normalize_obs       = True,
                     normalize_rewards   = True,
                     obs_clip            = (-10., 10.),
                     reward_clip         = (-10., 10.),
                     **self.kw_run_args)

from abc import ABC, abstractmethod
from abc import ABC, abstractmethod
from ppo_and_friends.ppo import PPO
from ppo_and_friends.testing import test_policy

from mpi4py import MPI
comm      = MPI.COMM_WORLD
num_procs = comm.Get_size()

class EnvironmentRunner(ABC):
    """
        A base class for running RL environments.
    """

    def __init__(self,
                 **kw_run_args):
        """
            Arguments:
                kw_run_args    Keywoard arguments for training.
        """
        self.kw_run_args = kw_run_args

    @abstractmethod
    def run(self):
        raise NotImplementedError

    def get_adjusted_ts_per_rollout(self, ts_per_rollout):
        if 'envs_per_proc' in self.kw_run_args:
            envs_per_proc = self.kw_run_args['envs_per_proc']
        else:
            envs_per_proc = 1
        return num_procs * ts_per_rollout * envs_per_proc

    def run_ppo(self,
                policy_settings,
                policy_mapping_fn,
                env_generator,
                device,
                test                  = False,
                explore_while_testing = False,
                num_timesteps         = 1_000_000,
                render_gif            = False,
                gif_fps               = 15,
                num_test_runs         = 1,
                **kw_args):

        """
            Run the PPO algorithm.
        """

        ppo = PPO(policy_settings   = policy_settings,
                  policy_mapping_fn = policy_mapping_fn,
                  env_generator     = env_generator,
                  device            = device,
                  test_mode         = test,
                  **kw_args)

        #
        # Pickling is a special case. It allows users to save the ppo class
        # for use elsewhere. So, we skip training if pickling is requested.
        #
        pickling = "pickle_class" in kw_args and kw_args["pickle_class"]

        if test:
            test_policy(ppo           = ppo,
                        explore       = explore_while_testing,
                        device        = device,
                        render_gif    = render_gif,
                        gif_fps       = gif_fps,
                        num_test_runs = num_test_runs,
                        **kw_args)

        elif not pickling:
            ppo.learn(num_timesteps)


class GymRunner(EnvironmentRunner):
    """
        A base class for running gym environments.
    """
    def get_gym_render_mode(self):
        """
            Get the render mode for a gym environment.
        """
        if self.kw_run_args["render"]:
            return "human"
        elif self.kw_run_args["render_gif"]:
            return "rgb_array"
        else:
            return None



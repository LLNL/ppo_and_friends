from abc import ABC, abstractmethod
from abc import ABC, abstractmethod
from ppo_and_friends.ppo import PPO
from ppo_and_friends.testing import test_policy
import argparse

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
        Parameters:
        -----------
        kw_run_args: dict
            Keywoard arguments for training.
        """
        self.kw_run_args = kw_run_args
        self.cli_args    = None

    def parse_extended_cli_args(self, args, namespace=None):
        """
        Parse an extended arg parser from the CLI. Users can define the
        'add_cli_args' method, which can be used to extend the CLI arg parser.
        Those args will then be added to the self.cli_args variable and
        accessible when defining the 'run' method.

        Parameters:
        -----------
        args: list
            A list of args to be passed to the runner's arg parser.
        namespace: argparse.Namespace
            An optional namespace to pass the parse_args.

        Returns:
        --------
        tuple:
            The parser and the parsed args.
        """
        parser        = argparse.ArgumentParser()
        parser        = self.add_cli_args(parser)
        self.cli_args = parser.parse_args(args=args, namespace=namespace)
        return parser, self.cli_args

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
        return parser

    @abstractmethod
    def run(self):
        raise NotImplementedError

    def run_ppo(self,
                policy_settings,
                policy_mapping_fn,
                env_generator,
                device,
                test                  = False,
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
        if "render" in self.kw_run_args and self.kw_run_args["render"]:
            return "human"
        elif "render_gif" in self.kw_run_args and self.kw_run_args["render_gif"]:
            return "rgb_array"
        else:
            return None

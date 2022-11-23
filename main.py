import gym
import torch
import random
import numpy as np
import argparse
from ppo_and_friends.environments.launchers import *
import os
from ppo_and_friends.utils.mpi_utils import rank_print
import shutil
from mpi4py import MPI

comm      = MPI.COMM_WORLD
rank      = comm.Get_rank()
num_procs = comm.Get_size()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--test", action="store_true",
        help="Test out an existing policy.")

    parser.add_argument("--test-explore", action="store_true",
        help="Enable exploration while testing. Note that this flag"
        "only has an effect while in test mode. Exploration is always"
        "enabled during training.")

    parser.add_argument("--num-test-runs", type=int, default=1,
        help="If used with --test, this will define the number of test "
        "iterations that are run. The min, max, and average scores will "
        "be reported.")

    parser.add_argument("--state-path", default="./",
        help="Where to save states and policy info. The data will be "
        "saved in a sub-directory named 'saved_states'.")

    parser.add_argument("--clobber", action="store_true",
        help="Clobber any existing saves associated with this environment.")

    parser.add_argument("--render", action="store_true",
        help="Render the environment at each step.")

    parser.add_argument("--render-gif", action="store_true",
        help="Render a gif when testing.")

    #TODO: let's also let users stop at an iteration rather than timestep.
    parser.add_argument("--num-timesteps", default=1000000, type=int,
        help="The number of timesteps to train for.")

    parser.add_argument("--random_seed", default=2, type=int,
        help="The random seed to use.")

    parser.add_argument("--envs-per-proc", default=1, type=int,
        help="The number of environment instances each processor should have.")

    parser.add_argument("--allow-mpi-gpu", action="store_true",
        help="By default, GPUs are only used if the we're training on a single "
        "processor, as this is very effecient when using MLPs. This flag will "
        "allow GPUs to be used with multiple processors, which is useful when "
        "using networks that are very slow on CPUs (like convolutions).")

    parser.add_argument("--force-deterministic", action="store_true",
        help="Tell PyTorch to only use deterministic algorithms.")

    parser.add_argument("--environment", "-e", type=str, required=True,
        help="Which environment should we train or test?",
        choices=["CartPole",
                 "Pendulum",
                 "Acrobot",
                 "MountainCar",
                 "MountainCarContinuous",
                 "LunarLander",
                 "LunarLanderContinuous",
                 "BipedalWalker",
                 "BipedalWalkerHardcore",
                 "BreakoutPixels",
                 "BreakoutRAM",
                 "InvertedPendulum",
                 "InvertedDoublePendulum",
                 "Ant",
                 "Walker2D",
                 "Hopper",
                 "Swimmer",
                 "HalfCheetah",
                 "HumanoidStandUp",
                 "Humanoid",
                 "RobotWarehouseTiny",
                 "RobotWarehouseSmall",
                 "LevelBasedForaging",
                 "PressurePlate",
                 "AbmarlMaze",
                 "AirstrikerRAM",
                 "BinaryCartPole",
                 "BinaryLunarLander",
                 ])

    args               = parser.parse_args()
    test               = args.test
    test_explore       = args.test_explore
    random_seed        = args.random_seed + rank
    num_test_runs      = args.num_test_runs
    env_name           = args.environment
    state_path         = os.path.join(args.state_path, "saved_states", env_name)
    clobber            = args.clobber
    render             = args.render
    render_gif         = args.render_gif
    num_timesteps      = args.num_timesteps
    force_determinism  = args.force_deterministic
    envs_per_proc      = args.envs_per_proc
    allow_mpi_gpu      = args.allow_mpi_gpu

    if render and render_gif:
        msg  = "ERROR: render and render_gif are both enabled, "
        msg += "but they cannot be used simultaneously."
        rank_print(msg)
        comm.Abort()

    if render_gif and not test:
        msg = "ERROR: render_gif is only allowed when testing."
        rank_print(msg)
        comm.Abort()

    if num_test_runs > 1 and not test:
        msg = "ERROR: --num_test_runs can only be used with --test."
        rank_print(msg)
        comm.Abort()

    #
    # Set random seeds (this doesn't guarantee reproducibility, but it should
    # help).
    #
    torch.manual_seed(random_seed)
    random.seed(random_seed)
    np.random.seed(random_seed)

    if force_determinism:
        torch.use_deterministic_algorithms(True)

    load_state = not clobber or test

    if clobber and rank == 0:
        if os.path.exists(state_path):
            shutil.rmtree(state_path)
    comm.barrier()

    if (torch.cuda.is_available() and
        not test and
        (num_procs == 1 or allow_mpi_gpu)):

        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    rank_print("Using device: {}".format(device))
    rank_print("Number of processors: {}".format(num_procs))
    rank_print("Number of environments per processor: {}".format(envs_per_proc))

    #
    # Launch PPO.
    #
    class_name     = "{}Launcher".format(env_name)
    launcher_class = getattr(sys.modules[__name__], class_name)

    launcher = launcher_class(
        random_seed           = random_seed,
        state_path            = state_path,
        load_state            = load_state,
        render                = render,
        render_gif            = render_gif,
        num_timesteps         = num_timesteps,
        device                = device,
        envs_per_proc         = envs_per_proc,
        test                  = test,
        explore_while_testing = test_explore,
        num_test_runs         = num_test_runs)

    launcher.launch()

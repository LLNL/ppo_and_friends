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
    parser.add_argument("--num_test_runs", type=int, default=1,
        help="If used with --test, this will define the number of test "
        "iterations that are run. The min, max, and average scores will "
        "be reported.")
    parser.add_argument("--state_path", default="./",
        help="Where to save states and policy info. The data will be "
        "saved in a sub-directory named 'saved_states'.")
    parser.add_argument("--clobber", action="store_true",
        help="Clobber any existing saves associated with this environment.")
    parser.add_argument("--render", action="store_true",
        help="Render the environment at each step.")
    parser.add_argument("--num_timesteps", default=10000000, type=int,
        help="The number of timesteps to train for.")
    parser.add_argument("--random_seed", default=2, type=int,
        help="The random seed to use.")
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
                 "AssaultRAM",
                 "AssaultPixels",
                 "BreakoutPixels",
                 "BreakoutRAM",
                 "InvertedPendulum",
                 "InvertedDoublePendulum",
                 "Ant",
                 "Walker2d",
                 "Hopper",
                 "Swimmer",
                 "HalfCheetah",
                 "HumanoidStandup",
                 "Humanoid"])

    args              = parser.parse_args()
    test              = args.test
    random_seed       = args.random_seed + rank
    num_test_runs     = args.num_test_runs
    env_name          = args.environment
    state_path        = os.path.join(args.state_path, "saved_states", env_name)
    clobber           = args.clobber
    render            = args.render
    num_timesteps     = args.num_timesteps
    force_determinism = args.force_deterministic

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

    if torch.cuda.is_available() and not test and num_procs == 1:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    rank_print("Using device: {}".format(device))
    rank_print("Number of processors: {}".format(num_procs))

    #TODO: we can probably simplify this here.
    if env_name == "CartPole":
        cartpole_ppo(
            random_seed   = random_seed,
            state_path    = state_path,
            load_state    = load_state,
            render        = render,
            num_timesteps = num_timesteps,
            device        = device,
            test          = test,
            num_test_runs = num_test_runs)

    elif env_name == "Pendulum":
        pendulum_ppo(
            random_seed   = random_seed,
            state_path    = state_path,
            load_state    = load_state,
            render        = render,
            num_timesteps = num_timesteps,
            device        = device,
            test          = test,
            num_test_runs = num_test_runs)

    elif env_name == "LunarLander":
        lunar_lander_ppo(
            random_seed   = random_seed,
            state_path    = state_path,
            load_state    = load_state,
            render        = render,
            num_timesteps = num_timesteps,
            device        = device,
            test          = test,
            num_test_runs = num_test_runs)

    elif env_name == "LunarLanderContinuous":
        lunar_lander_continuous_ppo(
            random_seed   = random_seed,
            state_path    = state_path,
            load_state    = load_state,
            render        = render,
            num_timesteps = num_timesteps,
            device        = device,
            test          = test,
            num_test_runs = num_test_runs)

    elif env_name == "MountainCar":
        mountain_car_ppo(
            random_seed   = random_seed,
            state_path    = state_path,
            load_state    = load_state,
            render        = render,
            num_timesteps = num_timesteps,
            device        = device,
            test          = test,
            num_test_runs = num_test_runs)

    elif env_name == "MountainCarContinuous":
        mountain_car_continuous_ppo(
            random_seed   = random_seed,
            state_path    = state_path,
            load_state    = load_state,
            render        = render,
            num_timesteps = num_timesteps,
            device        = device,
            test          = test,
            num_test_runs = num_test_runs)

    elif env_name == "Acrobot":
        acrobot_ppo(
            random_seed   = random_seed,
            state_path    = state_path,
            load_state    = load_state,
            render        = render,
            num_timesteps = num_timesteps,
            device        = device,
            test          = test,
            num_test_runs = num_test_runs)

    elif env_name == "AssaultRAM":
        assault_ram_ppo(
            random_seed   = random_seed,
            state_path    = state_path,
            load_state    = load_state,
            render        = render,
            num_timesteps = num_timesteps,
            device        = device,
            test          = test,
            num_test_runs = num_test_runs)

    elif env_name == "AssaultPixels":
        assault_pixels_ppo(
            random_seed   = random_seed,
            state_path    = state_path,
            load_state    = load_state,
            render        = render,
            num_timesteps = num_timesteps,
            device        = device,
            test          = test,
            num_test_runs = num_test_runs)

    elif env_name == "BreakoutPixels":
        breakout_pixels_ppo(
            random_seed   = random_seed,
            state_path    = state_path,
            load_state    = load_state,
            render        = render,
            num_timesteps = num_timesteps,
            device        = device,
            test          = test,
            num_test_runs = num_test_runs)

    elif env_name == "BreakoutRAM":
        breakout_ram_ppo(
            random_seed   = random_seed,
            state_path    = state_path,
            load_state    = load_state,
            render        = render,
            num_timesteps = num_timesteps,
            device        = device,
            test          = test,
            num_test_runs = num_test_runs)

    elif env_name == "BipedalWalker":
        bipedal_walker_ppo(
            random_seed   = random_seed,
            state_path    = state_path,
            load_state    = load_state,
            render        = render,
            num_timesteps = num_timesteps,
            device        = device,
            test          = test,
            num_test_runs = num_test_runs)

    elif env_name == "BipedalWalkerHardcore":
        bipedal_walker_hardcore_ppo(
            random_seed   = random_seed,
            state_path    = state_path,
            load_state    = load_state,
            render        = render,
            num_timesteps = num_timesteps,
            device        = device,
            test          = test,
            num_test_runs = num_test_runs)

    elif env_name == "InvertedPendulum":
        inverted_pendulum_ppo(
            random_seed   = random_seed,
            state_path    = state_path,
            load_state    = load_state,
            render        = render,
            num_timesteps = num_timesteps,
            device        = device,
            test          = test,
            num_test_runs = num_test_runs)

    elif env_name == "InvertedDoublePendulum":
        inverted_double_pendulum_ppo(
            random_seed   = random_seed,
            state_path    = state_path,
            load_state    = load_state,
            render        = render,
            num_timesteps = num_timesteps,
            device        = device,
            test          = test,
            num_test_runs = num_test_runs)

    elif env_name == "Ant":
        ant_ppo(
            random_seed   = random_seed,
            state_path    = state_path,
            load_state    = load_state,
            render        = render,
            num_timesteps = num_timesteps,
            device        = device,
            test          = test,
            num_test_runs = num_test_runs)

    elif env_name == "Walker2d":
        walker2d_ppo(
            random_seed   = random_seed,
            state_path    = state_path,
            load_state    = load_state,
            render        = render,
            num_timesteps = num_timesteps,
            device        = device,
            test          = test,
            num_test_runs = num_test_runs)

    elif env_name == "Hopper":
        hopper_ppo(
            random_seed   = random_seed,
            state_path    = state_path,
            load_state    = load_state,
            render        = render,
            num_timesteps = num_timesteps,
            device        = device,
            test          = test,
            num_test_runs = num_test_runs)

    elif env_name == "Swimmer":
        swimmer_ppo(
            random_seed   = random_seed,
            state_path    = state_path,
            load_state    = load_state,
            render        = render,
            num_timesteps = num_timesteps,
            device        = device,
            test          = test,
            num_test_runs = num_test_runs)

    elif env_name == "HalfCheetah":
        half_cheetah_ppo(
            random_seed   = random_seed,
            state_path    = state_path,
            load_state    = load_state,
            render        = render,
            num_timesteps = num_timesteps,
            device        = device,
            test          = test,
            num_test_runs = num_test_runs)

    elif env_name == "Humanoid":
        humanoid_ppo(
            random_seed   = random_seed,
            state_path    = state_path,
            load_state    = load_state,
            render        = render,
            num_timesteps = num_timesteps,
            device        = device,
            test          = test,
            num_test_runs = num_test_runs)

    elif env_name == "HumanoidStandup":
        humanoid_stand_up_ppo(
            random_seed   = random_seed,
            state_path    = state_path,
            load_state    = load_state,
            render        = render,
            num_timesteps = num_timesteps,
            device        = device,
            test          = test,
            num_test_runs = num_test_runs)

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

    parser.add_argument("--render_gif", action="store_true",
        help="Render a gif when testing.")

    #TODO: let's also let users stop at an iteration rather than timestep.
    parser.add_argument("--num_timesteps", default=10000000, type=int,
        help="The number of timesteps to train for.")

    parser.add_argument("--random_seed", default=2, type=int,
        help="The random seed to use.")

    parser.add_argument("--envs_per_proc", default=1, type=int,
        help="The number of environment instances each processor should have.")

    parser.add_argument("--allow_mpi_gpu", action="store_true",
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
                 "Walker2d",
                 "Hopper",
                 "Swimmer",
                 "HalfCheetah",
                 "HumanoidStandup",
                 "Humanoid",
                 "RobotWarehouseTiny",
                 "RobotWarehouseSmall",
                 "LevelBasedForaging",])

    args              = parser.parse_args()
    test              = args.test
    random_seed       = args.random_seed + rank
    num_test_runs     = args.num_test_runs
    env_name          = args.environment
    state_path        = os.path.join(args.state_path, "saved_states", env_name)
    clobber           = args.clobber
    render            = args.render
    render_gif        = args.render_gif
    num_timesteps     = args.num_timesteps
    force_determinism = args.force_deterministic
    envs_per_proc     = args.envs_per_proc
    allow_mpi_gpu     = args.allow_mpi_gpu

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
    # Good distribution for CartPole:
    # 4 processors and 2 envs per processor takes ~20->30 iterations.
    #
    if env_name == "CartPole":
        cartpole_ppo(
            random_seed   = random_seed,
            state_path    = state_path,
            load_state    = load_state,
            render        = render,
            render_gif    = render_gif,
            num_timesteps = num_timesteps,
            device        = device,
            envs_per_proc = envs_per_proc,
            test          = test,
            num_test_runs = num_test_runs)

    #
    # Good distribution for Pendulum:
    # 2 processors and 2 envs per processor.
    #
    elif env_name == "Pendulum":
        pendulum_ppo(
            random_seed   = random_seed,
            state_path    = state_path,
            load_state    = load_state,
            render        = render,
            render_gif    = render_gif,
            num_timesteps = num_timesteps,
            device        = device,
            envs_per_proc = envs_per_proc,
            test          = test,
            num_test_runs = num_test_runs)

    #
    # Good distribution for LunarLander:
    # 2 processors and 2 envs per processor.
    #
    elif env_name == "LunarLander":
        lunar_lander_ppo(
            random_seed   = random_seed,
            state_path    = state_path,
            load_state    = load_state,
            render        = render,
            render_gif    = render_gif,
            num_timesteps = num_timesteps,
            device        = device,
            envs_per_proc = envs_per_proc,
            test          = test,
            num_test_runs = num_test_runs)

    #
    # Good distribution for LunarLander:
    # 2 processors and 2 envs per processor.
    #
    elif env_name == "LunarLanderContinuous":
        lunar_lander_continuous_ppo(
            random_seed   = random_seed,
            state_path    = state_path,
            load_state    = load_state,
            render        = render,
            render_gif    = render_gif,
            num_timesteps = num_timesteps,
            device        = device,
            envs_per_proc = envs_per_proc,
            test          = test,
            num_test_runs = num_test_runs)

    #
    # Good distribution for MountainCar:
    # 2 processors and 3 envs per processor takes about 6 min.
    #
    elif env_name == "MountainCar":
        mountain_car_ppo(
            random_seed   = random_seed,
            state_path    = state_path,
            load_state    = load_state,
            render        = render,
            render_gif    = render_gif,
            num_timesteps = num_timesteps,
            device        = device,
            envs_per_proc = envs_per_proc,
            test          = test,
            num_test_runs = num_test_runs)

    #
    # Good distribution for MountainCarContinuous:
    # 2 processors and 3 envs per processor.
    #
    elif env_name == "MountainCarContinuous":
        mountain_car_continuous_ppo(
            random_seed   = random_seed,
            state_path    = state_path,
            load_state    = load_state,
            render        = render,
            render_gif    = render_gif,
            num_timesteps = num_timesteps,
            device        = device,
            envs_per_proc = envs_per_proc,
            test          = test,
            num_test_runs = num_test_runs)

    #
    # Good distribution for Acrobot:
    # 2 processors and 1 envs per processor learns a good policy
    # in about 30 seconds to a minute.
    #
    elif env_name == "Acrobot":
        acrobot_ppo(
            random_seed   = random_seed,
            state_path    = state_path,
            load_state    = load_state,
            render        = render,
            render_gif    = render_gif,
            num_timesteps = num_timesteps,
            device        = device,
            envs_per_proc = envs_per_proc,
            test          = test,
            num_test_runs = num_test_runs)

    #
    # For Atari Pixel environments, I recommend using the --allow_mpi_gpu
    # so that you can leverage MPI for rollout speed and GPUs for training
    # convolutional networks. I haven't explored processor and env counts
    # too much, but 4 processors and 2 envs per proc worked well in
    # BreakoutPixels. Similar settings worked well for BreakoutRAM, but
    # you don't need to use the --allow_mpi_gpu flag.
    #
    elif env_name == "BreakoutPixels":
        breakout_pixels_ppo(
            random_seed   = random_seed,
            state_path    = state_path,
            load_state    = load_state,
            render        = render,
            render_gif    = render_gif,
            num_timesteps = num_timesteps,
            device        = device,
            envs_per_proc = envs_per_proc,
            test          = test,
            num_test_runs = num_test_runs)

    elif env_name == "BreakoutRAM":
        breakout_ram_ppo(
            random_seed   = random_seed,
            state_path    = state_path,
            load_state    = load_state,
            render        = render,
            render_gif    = render_gif,
            num_timesteps = num_timesteps,
            device        = device,
            envs_per_proc = envs_per_proc,
            test          = test,
            num_test_runs = num_test_runs)

    #
    # Good distribution for BipedalWalker:
    # 4 processors and 1 env per processor will solve the environment
    # in about 15->20 minutes. The environment is considered solved when
    # the average score over 100 test runs >= 300.
    #
    elif env_name == "BipedalWalker":
        bipedal_walker_ppo(
            random_seed   = random_seed,
            state_path    = state_path,
            load_state    = load_state,
            render        = render,
            render_gif    = render_gif,
            num_timesteps = num_timesteps,
            device        = device,
            envs_per_proc = envs_per_proc,
            test          = test,
            num_test_runs = num_test_runs)

    #
    # Good distribution for BipedalWalker:
    # 4 processors and 1 env per processor can solve the environment in
    # 4 or so hours on my system.
    #
    elif env_name == "BipedalWalkerHardcore":
        bipedal_walker_hardcore_ppo(
            random_seed   = random_seed,
            state_path    = state_path,
            load_state    = load_state,
            render        = render,
            render_gif    = render_gif,
            num_timesteps = num_timesteps,
            device        = device,
            envs_per_proc = envs_per_proc,
            test          = test,
            num_test_runs = num_test_runs)

    #
    # Good distribution for InvertedPendulum:
    # 2 processors and 2 envs per processor learns a good policy
    # in about 10 seconds of training.
    #
    elif env_name == "InvertedPendulum":
        inverted_pendulum_ppo(
            random_seed   = random_seed,
            state_path    = state_path,
            load_state    = load_state,
            render        = render,
            render_gif    = render_gif,
            num_timesteps = num_timesteps,
            device        = device,
            envs_per_proc = envs_per_proc,
            test          = test,
            num_test_runs = num_test_runs)

    #
    # Good distribution for InvertedPendulum:
    # 2 processors and 2 envs per processor learns a good policy
    # within a few minutes.
    #
    elif env_name == "InvertedDoublePendulum":
        inverted_double_pendulum_ppo(
            random_seed   = random_seed,
            state_path    = state_path,
            load_state    = load_state,
            render        = render,
            render_gif    = render_gif,
            num_timesteps = num_timesteps,
            device        = device,
            envs_per_proc = envs_per_proc,
            test          = test,
            num_test_runs = num_test_runs)

    #
    # Good distribution for Ant:
    # 2 processors and 2 envs per processor learns an excellent
    # policy in ~10 min.
    #
    elif env_name == "Ant":
        ant_ppo(
            random_seed   = random_seed,
            state_path    = state_path,
            load_state    = load_state,
            render        = render,
            render_gif    = render_gif,
            num_timesteps = num_timesteps,
            device        = device,
            envs_per_proc = envs_per_proc,
            test          = test,
            num_test_runs = num_test_runs)

    #
    # Good distribution for Walker:
    # 2 processors and 2 envs per processor works well.
    #
    elif env_name == "Walker2d":
        walker2d_ppo(
            random_seed   = random_seed,
            state_path    = state_path,
            load_state    = load_state,
            render        = render,
            render_gif    = render_gif,
            num_timesteps = num_timesteps,
            device        = device,
            envs_per_proc = envs_per_proc,
            test          = test,
            num_test_runs = num_test_runs)

    #
    # Good distribution for Hopper:
    # Hopper is oddly sensitive to the trajectory lengths. It will learn
    # to go a bit beyond your max trajectory length very well, but will
    # often fall after that. The max ts per rollout are set to 2048 in
    # this environment, so I recommend using 2 processors or 2 environments
    # per proc but not both.
    #
    elif env_name == "Hopper":
        hopper_ppo(
            random_seed   = random_seed,
            state_path    = state_path,
            load_state    = load_state,
            render        = render,
            render_gif    = render_gif,
            num_timesteps = num_timesteps,
            device        = device,
            envs_per_proc = envs_per_proc,
            test          = test,
            num_test_runs = num_test_runs)

    #
    # Good distribution for Swimmer:
    # 2 processors and 2 envs per processor works well.
    #
    elif env_name == "Swimmer":
        swimmer_ppo(
            random_seed   = random_seed,
            state_path    = state_path,
            load_state    = load_state,
            render        = render,
            render_gif    = render_gif,
            num_timesteps = num_timesteps,
            device        = device,
            envs_per_proc = envs_per_proc,
            test          = test,
            num_test_runs = num_test_runs)

    #
    # Good distribution for HalfCheetah:
    # 2 processors and 2 envs per processor will learn an excellent
    # policy in about 2 minutes.
    #
    elif env_name == "HalfCheetah":
        half_cheetah_ppo(
            random_seed   = random_seed,
            state_path    = state_path,
            load_state    = load_state,
            render        = render,
            render_gif    = render_gif,
            num_timesteps = num_timesteps,
            device        = device,
            envs_per_proc = envs_per_proc,
            test          = test,
            num_test_runs = num_test_runs)

    #
    # Good distribution for Humanoid:
    # 2 processors and 2 envs per processor learns an excellent policy in
    # 40 minutes or less.
    #
    elif env_name == "Humanoid":
        humanoid_ppo(
            random_seed   = random_seed,
            state_path    = state_path,
            load_state    = load_state,
            render        = render,
            render_gif    = render_gif,
            num_timesteps = num_timesteps,
            device        = device,
            envs_per_proc = envs_per_proc,
            test          = test,
            num_test_runs = num_test_runs)

    elif env_name == "HumanoidStandup":
        humanoid_stand_up_ppo(
            random_seed   = random_seed,
            state_path    = state_path,
            load_state    = load_state,
            render        = render,
            render_gif    = render_gif,
            num_timesteps = num_timesteps,
            device        = device,
            envs_per_proc = envs_per_proc,
            test          = test,
            num_test_runs = num_test_runs)

    elif env_name == "RobotWarehouseTiny":
        robot_warehouse_tiny(
            random_seed   = random_seed,
            state_path    = state_path,
            load_state    = load_state,
            render        = render,
            render_gif    = render_gif,
            num_timesteps = num_timesteps,
            device        = device,
            envs_per_proc = envs_per_proc,
            test          = test,
            num_test_runs = num_test_runs)

    elif env_name == "RobotWarehouseSmall":
        robot_warehouse_small(
            random_seed   = random_seed,
            state_path    = state_path,
            load_state    = load_state,
            render        = render,
            render_gif    = render_gif,
            num_timesteps = num_timesteps,
            device        = device,
            envs_per_proc = envs_per_proc,
            test          = test,
            num_test_runs = num_test_runs)

    elif env_name == "LevelBasedForaging":
        level_based_foraging(
            random_seed   = random_seed,
            state_path    = state_path,
            load_state    = load_state,
            render        = render,
            render_gif    = render_gif,
            num_timesteps = num_timesteps,
            device        = device,
            envs_per_proc = envs_per_proc,
            test          = test,
            num_test_runs = num_test_runs)

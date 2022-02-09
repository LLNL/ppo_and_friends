import gym
import torch
import random
import numpy as np
import argparse
from ppo_and_friends.environments.launchers import *
import os


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--num_test_runs", type=int, default=1)
    parser.add_argument("--state_path", default="")
    parser.add_argument("--clobber", action="store_true")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--num_timesteps", default=10000000, type=int)
    parser.add_argument("--random_seed", default=2, type=int)
    parser.add_argument("--environment", "-e", type=str, required=True,
        choices=["CartPole",
                 "Pendulum",
                 "Acrobot",
                 "MountainCar",
                 "MountainCarContinuous",
                 "LunarLander",
                 "LunarLanderContinuous",
                 "BipedalWalker",
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

    args          = parser.parse_args()
    test          = args.test
    random_seed   = args.random_seed
    num_test_runs = args.num_test_runs
    env_name      = args.environment
    state_path    = os.path.join(args.state_path, "saved_states", env_name)
    clobber       = args.clobber
    render        = args.render
    num_timesteps = args.num_timesteps

    #
    # Set random seeds (this doesn't guarantee reproducibility, but it should
    # help).
    #
    torch.manual_seed(random_seed)
    random.seed(random_seed)
    np.random.seed(random_seed)

    load_state = not clobber or test

    if torch.cuda.is_available() and not test:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print("Using device: {}".format(device))

    #TODO: we can probably simplify this here.
    if env_name == "CartPole":
        cartpole_ppo(
            state_path    = state_path,
            load_state    = load_state,
            render        = render,
            num_timesteps = num_timesteps,
            device        = device,
            test          = test,
            num_test_runs = num_test_runs)

    elif env_name == "Pendulum":
        pendulum_ppo(
            state_path    = state_path,
            load_state    = load_state,
            render        = render,
            num_timesteps = num_timesteps,
            device        = device,
            test          = test,
            num_test_runs = num_test_runs)

    elif env_name == "LunarLander":
        lunar_lander_ppo(
            state_path    = state_path,
            load_state    = load_state,
            render        = render,
            num_timesteps = num_timesteps,
            device        = device,
            test          = test,
            num_test_runs = num_test_runs)

    elif env_name == "LunarLanderContinuous":
        lunar_lander_continuous_ppo(
            state_path    = state_path,
            load_state    = load_state,
            render        = render,
            num_timesteps = num_timesteps,
            device        = device,
            test          = test,
            num_test_runs = num_test_runs)

    elif env_name == "MountainCar":
        mountain_car_ppo(
            state_path    = state_path,
            load_state    = load_state,
            render        = render,
            num_timesteps = num_timesteps,
            device        = device,
            test          = test,
            num_test_runs = num_test_runs)

    elif env_name == "MountainCarContinuous":
        mountain_car_continuous_ppo(
            state_path    = state_path,
            load_state    = load_state,
            render        = render,
            num_timesteps = num_timesteps,
            device        = device,
            test          = test,
            num_test_runs = num_test_runs)

    elif env_name == "Acrobot":
        acrobot_ppo(
            state_path    = state_path,
            load_state    = load_state,
            render        = render,
            num_timesteps = num_timesteps,
            device        = device,
            test          = test,
            num_test_runs = num_test_runs)

    elif env_name == "AssaultRAM":
        assault_ram_ppo(
            state_path    = state_path,
            load_state    = load_state,
            render        = render,
            num_timesteps = num_timesteps,
            device        = device,
            test          = test,
            num_test_runs = num_test_runs)

    elif env_name == "AssaultPixels":
        assault_pixels_ppo(
            state_path    = state_path,
            load_state    = load_state,
            render        = render,
            num_timesteps = num_timesteps,
            device        = device,
            test          = test,
            num_test_runs = num_test_runs)

    elif env_name == "BreakoutPixels":
        breakout_pixels_ppo(
            state_path    = state_path,
            load_state    = load_state,
            render        = render,
            num_timesteps = num_timesteps,
            device        = device,
            test          = test,
            num_test_runs = num_test_runs)

    elif env_name == "BreakoutRAM":
        breakout_ram_ppo(
            state_path    = state_path,
            load_state    = load_state,
            render        = render,
            num_timesteps = num_timesteps,
            device        = device,
            test          = test,
            num_test_runs = num_test_runs)

    elif env_name == "BipedalWalker":
        bipedal_walker_ppo(
            state_path    = state_path,
            load_state    = load_state,
            render        = render,
            num_timesteps = num_timesteps,
            device        = device,
            test          = test,
            num_test_runs = num_test_runs)

    elif env_name == "InvertedPendulum":
        inverted_pendulum_ppo(
            state_path    = state_path,
            load_state    = load_state,
            render        = render,
            num_timesteps = num_timesteps,
            device        = device,
            test          = test,
            num_test_runs = num_test_runs)

    elif env_name == "InvertedDoublePendulum":
        inverted_double_pendulum_ppo(
            state_path    = state_path,
            load_state    = load_state,
            render        = render,
            num_timesteps = num_timesteps,
            device        = device,
            test          = test,
            num_test_runs = num_test_runs)

    elif env_name == "Ant":
        ant_ppo(
            state_path    = state_path,
            load_state    = load_state,
            render        = render,
            num_timesteps = num_timesteps,
            device        = device,
            test          = test,
            num_test_runs = num_test_runs)

    elif env_name == "Walker2d":
        walker2d_ppo(
            state_path    = state_path,
            load_state    = load_state,
            render        = render,
            num_timesteps = num_timesteps,
            device        = device,
            test          = test,
            num_test_runs = num_test_runs)

    elif env_name == "Hopper":
        hopper_ppo(
            state_path    = state_path,
            load_state    = load_state,
            render        = render,
            num_timesteps = num_timesteps,
            device        = device,
            test          = test,
            num_test_runs = num_test_runs)

    elif env_name == "Swimmer":
        swimmer_ppo(
            state_path    = state_path,
            load_state    = load_state,
            render        = render,
            num_timesteps = num_timesteps,
            device        = device,
            test          = test,
            num_test_runs = num_test_runs)

    elif env_name == "HalfCheetah":
        half_cheetah_ppo(
            state_path    = state_path,
            load_state    = load_state,
            render        = render,
            num_timesteps = num_timesteps,
            device        = device,
            test          = test,
            num_test_runs = num_test_runs)

    elif env_name == "Humanoid":
        humanoid_ppo(
            state_path    = state_path,
            load_state    = load_state,
            render        = render,
            num_timesteps = num_timesteps,
            device        = device,
            test          = test,
            num_test_runs = num_test_runs)

    elif env_name == "HumanoidStandup":
        humanoid_stand_up_ppo(
            state_path    = state_path,
            load_state    = load_state,
            render        = render,
            num_timesteps = num_timesteps,
            device        = device,
            test          = test,
            num_test_runs = num_test_runs)

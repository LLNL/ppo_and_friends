import gym
import torch
import argparse
from environments import *
import os


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--num_test_runs", type=int, default=1)
    parser.add_argument("--state_path", default="")
    parser.add_argument("--clobber", action="store_true")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--num_timesteps", default=1000000, type=int)
    parser.add_argument("--environment", "-e", type=str, required=True,
        choices=["CartPole", "CartPolePixels", "Pendulum", "LunarLander",
                 "MountainCar", "MountainCarContinuous", "Acrobot",
                 "AssaultRAM", "AssaultPixels",
                 "BreakoutPixels", "BreakoutRAM",
                 "LunarLanderContinuous",
                 "BipedalWalker",
                 "InvertedPendulum", "Ant"])

    args          = parser.parse_args()
    test          = args.test
    num_test_runs = args.num_test_runs
    env_name      = args.environment
    state_path    = os.path.join(args.state_path, "saved_states", env_name)
    clobber       = args.clobber
    render        = args.render
    num_timesteps = args.num_timesteps

    load_state = not clobber or test

    if torch.cuda.is_available() and not test:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    if env_name == "CartPole":
        cartpole_ppo(
            state_path    = state_path,
            load_state    = load_state,
            render        = render,
            num_timesteps = num_timesteps,
            device        = device,
            test          = test,
            num_test_runs = num_test_runs)

    elif env_name == "CartPolePixels":
        cartpole_pixels_ppo(
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

    elif env_name == "Ant":
        ant_ppo(
            state_path    = state_path,
            load_state    = load_state,
            render        = render,
            num_timesteps = num_timesteps,
            device        = device,
            test          = test,
            num_test_runs = num_test_runs)

from ppo import PPO
import gym
import torch
import argparse
from testing import test_policy

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--model_path", default="./saved_models")
    parser.add_argument("--load_state", action="store_true")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--action_type", default="continuous",
        choices=["continuous", "discrete"])
    parser.add_argument("--num_timesteps", default=500000, type=int)

    args          = parser.parse_args()
    test          = args.test
    model_path    = args.model_path
    load_state    = args.load_state or test
    render        = args.render
    action_type   = args.action_type
    num_timesteps = args.num_timesteps

    if torch.cuda.is_available() and not test:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    #env = gym.make('Pendulum-v1')
    env = gym.make('CartPole-v0')

    ppo = PPO(env          = env,
              device       = device,
              action_type  = action_type,
              render       = render,
              load_weights = load_state,
              model_path   = model_path)

    if test:
        test_policy(ppo.actor, env, render, device, action_type)
    else: 
        ppo.learn(num_timesteps)


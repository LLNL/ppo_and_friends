import gym
import argparse
import json
import os
import numpy as np
from itertools import count
from PIL import Image
import torch
import torch.optim as optim
from utils import StateExperience
from networks import LinearDQN, Conv1dDQN, Linear2DQN, SimpleDQN, LinearDQN3
from networks import ConvDQN
from utils import CartPoleEnvManager
from utils import get_moving_average
from utils import EpsilonGreedyStrategy
from utils import ReplayMemory
from utils import extract_state_tensors
from utils import QValues
import torch.nn.functional as t_functional
import torch.nn as nn




if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", required=True,
        choices=["linear", "conv", "linear2", "linear3", "simple"])

    args = parser.parse_args()

    model_type = args.model_type

    save_path     = "from_state_save"
    batch_size    = 128
    gamma         = 0.95
    eps_max       = 1.
    eps_min       = 0.01
    eps_decay     = 0.001
    target_update = 30
    memory_size   = 100000
    lr            = 0.001
    num_episodes  = 2000
    top_score     = 199
    max_avg_drop  = -20
    grace_period  = 50
    grace_dist    = 50
    
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    
    env          = gym.make("CartPole-v0")
    memory       = ReplayMemory(memory_size)
    episode_durations = []

    if model_type == "linear":
        policy_net = LinearDQN(
            "LinearDQNPolicy",
            env.observation_space.shape[0]).to(device)
        
        target_net = LinearDQN(
            "LinearDQNTarget",
            env.observation_space.shape[0]).to(device)

    elif model_type == "linear2":
        policy_net = Linear2DQN(
            "Linear2DQNPolicy",
            env.observation_space.shape[0]).to(device)
        
        target_net = Linear2DQN(
            "Linear2DQNTarget",
            env.observation_space.shape[0]).to(device)

    elif model_type == "linear3":
        policy_net = LinearDQN3(
            "LinearDQN3Policy",
            env.observation_space.shape[0]).to(device)
        
        target_net = LinearDQN3(
            "LinearDQNTarget3",
            env.observation_space.shape[0]).to(device)

    elif model_type == "simple":
        policy_net = SimpleDQN(
            "SimpleDQN",
            env.observation_space.shape[0]).to(device)
        
        target_net = SimpleDQN(
            "SimpleDQN",
            env.observation_space.shape[0]).to(device)

    elif model_type == "conv":
        policy_net = Conv1dDQN(
            "LinearDQNPolicy",
            env.observation_space.shape[0]).to(device)
        
        target_net = Conv1dDQN(
            "LinearDQNTarget",
            env.observation_space.shape[0]).to(device)

    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()
    
    episode_info = {}
    episode_info["current_episode"]     = 0
    episode_info["epsilon"]             = eps_max
    episode_info["episode_durations"]   = episode_durations
    episode_info["top_average"]         = 0.0
    episode_info["top_average_episode"] = 0.0
    info_file = os.path.join(save_path, "episode_info.json")
    
    if os.path.exists(os.path.join(save_path, target_net.name + ".model")):
        print("Loading weights from saved model.")
    
        policy_net.load(save_path)
        target_net.load(save_path)
        memory.load(save_path)
    
        if os.path.exists(info_file):
            with open(info_file, "r") as in_f:
                episode_info = json.load(in_f)
    else:
        os.makedirs(save_path, exist_ok=True)
    
    optimizer = optim.Adam(params = policy_net.parameters(), lr = lr)
    
    starting_episode   = episode_info["current_episode"]
    stopping_episode   = starting_episode + num_episodes
    episode_durations  = episode_info["episode_durations"]
    epsilon            = episode_info["epsilon"]
    top_average        = episode_info["top_average"]
    top_avg_episode    = episode_info["top_average_episode"]

    loss_func = nn.SmoothL1Loss()
    
    msg  = "Starting from episode {}, "
    msg += "using epsilon {}".format(starting_episode, epsilon)
    print(msg)

    bail = False
    
    for episode_idx in range(starting_episode, stopping_episode):

        state = torch.Tensor([env.reset()])
    
        for ts in count():
            if np.random.random() > epsilon:
                with torch.no_grad():
                    action = policy_net(state.to(device)).to(torch.device('cpu'))
                action = torch.argmax(action, dim=1).long()
            else:
                action = torch.Tensor([np.random.choice(2)]).long()

            next_state, reward, done, _ = env.step(action.item())

            next_state = torch.Tensor([next_state])
            reward     = torch.Tensor([reward])
            done       = torch.Tensor([done]).bool()

            exp = StateExperience(state, action, next_state, reward, done)
            memory.push(exp)

            state = next_state.clone()
    
            if memory.can_provide_sample(batch_size):
                experiences = memory.sample(batch_size)

                states, actions, rewards, next_states, dones = \
                    extract_state_tensors(experiences, device)

                indices = torch.arange(batch_size).long().to(device)
                current_q_values     = policy_net.forward(states)
                current_q_values     = current_q_values[indices, actions]

                next_pred            = policy_net.forward(next_states)
                next_actions         = torch.argmax(next_pred, dim=1)
                next_q_values        = target_net.forward(next_states)
                next_q_values        = next_q_values[indices, next_actions]
                next_q_values[dones] = 0.

                q_target = rewards + (gamma * next_q_values)

                loss = loss_func(current_q_values, q_target)
    
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epsilon = eps_min + ((1.0 - eps_min) * np.exp(-1. * eps_decay * episode_idx))
    
            if done:
                episode_durations.append(ts)
                avg = float(get_moving_average(100, episode_durations)[-1])
                print("\nEpisode, epsilon: {}, {}".format(episode_idx, epsilon))
                print("    Moving average: {}".format(avg))

                top_reached = avg >= top_score
                avg_diff    = avg - top_average

                #
                # If our average is dropping too much, we're probably
                # forgetting too much. Let's bail.
                #
                if avg_diff <= max_avg_drop:
                    if grace_dist > 0:
                        grace_dist -= 1
                    else:
                        print("    Max average drop reached.")
                        print("    Re-loading top performing weights.")
                        policy_net.load(save_path)
                        target_net.load(save_path)
                        grace_dist = grace_period

                elif avg_diff >= 5.0 or top_reached:
                    print("    Saving model and memory.")
                    target_net.save(save_path)
                    policy_net.save(save_path)
                    memory.save(save_path)
                    top_average = avg
                    top_avg_episode = episode_idx

                   with open(info_file, "w") as out_f:
                       json.dump(episode_info, out_f)

                if top_reached:
                    print("Top average reached!!")
                    bail = True

                break

        if bail:
            break
    
        if episode_idx % target_update == 0:
            target_net.load_state_dict(policy_net.state_dict())

    print("---------------------------------------------")
    msg  = "Top moving average: {}, reached at ".format(top_average)
    msg += "episode {}.".format(top_avg_episode)
    print(msg)
    top_count = episode_durations.count(top_score)
    print("Top score reached {} times.".format(top_count))

    episode_info["current_episode"] = episode_idx
    episode_info["epsilon"]         = epsilon
    episode_info["top_average"]     = top_average
    episode_info["top_average_episode"] = top_avg_episode
    
    with open(info_file, "w") as out_f:
        json.dump(episode_info, out_f)
    
    env.close()

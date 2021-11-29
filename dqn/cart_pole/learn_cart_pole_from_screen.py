import gym
import argparse
import json
import os
import numpy as np
from itertools import count
from PIL import Image
import torch
import torch.optim as optim
from utils import ScreenExperience
from networks import LinearDQN, Linear2DQN
from networks import ConvDQN, ConvDQN2
from agents import DQNAgent
from utils import CartPoleEnvManager
from utils import get_moving_average
from utils import EpsilonGreedyStrategy
from utils import ReplayMemory
from utils import extract_screen_tensors
from utils import QValues
import torch.nn.functional as t_functional


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", required=True,
        choices=["linear", "linear2", "conv", "conv2"])

    args       = parser.parse_args()
    model_type = args.model_type

    save_path     = "from_screen_save"
    batch_size    = 128
    gamma         = 0.95
    eps_max     = 1.
    eps_min      = 0.01
    eps_decay     = 0.001
    target_update = 30
    memory_size   = 100000
    lr            = 0.001
    num_episodes  = 1000
    top_score     = 199
    max_avg_drop  = -20
    grace_period  = 50
    grace_dist    = 50
    
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    
    env_manager  = CartPoleEnvManager(device)
    eps_strategy = EpsilonGreedyStrategy(eps_max, eps_min, eps_decay)
    agent        = DQNAgent(eps_strategy, env_manager.num_actions_available(), device)
    memory       = ReplayMemory(memory_size)
    save_every   = num_episodes
    episode_durations = []

    if model_type == "linear":
        in_length = env_manager.get_screen_height() *\
            env_manager.get_screen_width() * 3

        policy_net = LinearDQN(
            "LinearDQNPolicy",
            in_length).to(device)
        
        target_net = LinearDQN(
            "LinearDQNTarget",
            in_length).to(device)

    elif model_type == "linear2":
        in_length = env_manager.get_screen_height() *\
            env_manager.get_screen_width() * 3

        policy_net = Linear2DQN(
            "Linear2DQNPolicy",
            in_length).to(device)
        
        target_net = Linear2DQN(
            "Linear2DQNTarget",
            in_length).to(device)

    elif model_type == "conv":    
        policy_net = ConvDQN(
            "ConvDQNPolicy",
            env_manager.get_screen_height(), 
            env_manager.get_screen_width()).to(device)
        
        target_net = ConvDQN(
            "ConvDQNTarget",
            env_manager.get_screen_height(), 
            env_manager.get_screen_width()).to(device)

    elif model_type == "conv2":
        policy_net = ConvDQN2(
            "ConvDQN2Policy",
            env_manager.get_screen_height(), 
            env_manager.get_screen_width()).to(device)
        
        target_net = ConvDQN2(
            "ConvDQN2Target",
            env_manager.get_screen_height(), 
            env_manager.get_screen_width()).to(device)

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
    agent.current_step = starting_episode
    
    print("Starting from episode: {}".format(starting_episode))

    bail = False
    
    for episode_idx in range(starting_episode, stopping_episode):
        env_manager.reset()
        state = env_manager.get_screen_state()
    
        for ts in count():

            if np.random.random() > epsilon:
                with torch.no_grad():
                    action = policy_net(state.to(device)).to(torch.device('cpu'))
                action = torch.argmax(action, dim=1).long()
            else:
                action = torch.Tensor([np.random.choice(2)]).long()

            reward     = env_manager.take_action(action)
            next_state = env_manager.get_screen_state()
    
            memory.push(ScreenExperience(state, action, next_state, reward))
    
            state = next_state
    
            if memory.can_provide_sample(batch_size):
                experiences = memory.sample(batch_size)
    
                states, actions, rewards, next_states = \
                    extract_screen_tensors(experiences, device)
    
                current_q_values = QValues.get_current(policy_net, states, actions)
                next_actions     = torch.argmax(policy_net(next_states), dim=1)
    
                next_q_values    = QValues.get_next_from_screen(target_net,
                    next_states, next_actions)
                target_q_values  = (next_q_values * gamma) + rewards
    
                loss = t_functional.smooth_l1_loss(current_q_values, target_q_values.unsqueeze(1))
    
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epsilon = eps_min + ((1.0 - eps_min) * np.exp(-1. * eps_decay * episode_idx))
    
            if env_manager.done:
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
    
    env_manager.close()

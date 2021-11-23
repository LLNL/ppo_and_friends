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
from networks import ConvDQN
from agents import DQNAgent
from utils import CartPoleEnvManager
from utils import plot
from utils import EpsilonGreedyStrategy
from utils import ReplayMemory
from utils import extract_screen_tensors
from utils import QValues
import torch.nn.functional as t_functional


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", required=True,
        choices=["linear", "linear2", "conv"])

    args = parser.parse_args()

    model_type = args.model_type

    save_path     = "from_screen_save"
    batch_size    = 256
    gamma         = 0.999
    eps_start     = 1.
    eps_stop      = 0.01
    eps_decay     = 0.001
    target_update = 10
    memory_size   = 100000
    lr            = 0.001
    num_episodes  = 10
    
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    
    env_manager  = CartPoleEnvManager(device)
    eps_strategy = EpsilonGreedyStrategy(eps_start, eps_stop, eps_decay)
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

    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()
    
    episode_info = {}
    episode_info["current_episode"]   = 0
    episode_info["episode_durations"] =  episode_durations
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
    agent.current_step = starting_episode
    
    print("Starting from episode: {}".format(starting_episode))
    
    for episode_idx in range(starting_episode, stopping_episode):
        env_manager.reset()
        state = env_manager.get_screen_state()
    
        for ts in count():
            action     = agent.select_action(state, policy_net).to(device)
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
    
                #loss = t_functional.mse_loss(current_q_values, target_q_values.unsqueeze(1))
                loss = t_functional.smooth_l1_loss(current_q_values, target_q_values.unsqueeze(1))
    
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
    
            if env_manager.done:
                episode_durations.append(ts)
                plot(episode_durations, 100, episode_idx,
                    eps_strategy.get_exploration_rate(agent.current_step))
                break
    
        last_episode = episode_idx == stopping_episode - 1

        if episode_idx % save_every == 0 or last_episode:
            target_net.save(save_path)
            policy_net.save(save_path)
            memory.save(save_path)
    
        episode_info["current_episode"] = episode_idx
    
        with open(info_file, "w") as out_f:
            json.dump(episode_info, out_f)
    
        if episode_idx % target_update == 0:
            target_net.load_state_dict(policy_net.state_dict())
    
    env_manager.close()

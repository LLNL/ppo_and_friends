import numpy as np
import torch

class DQNAgent(object):

    def  __init__(self, eps_strategy, num_actions, device):

        self.current_step = 0
        self.eps_strategy = eps_strategy
        self.num_actions  = num_actions
        self.device       = device

    def select_action(self, state, policy_net):

        rate = self.eps_strategy.get_exploration_rate(self.current_step)
        self.current_step += 1

        if rate > np.random.random():
            action =  np.random.randint(self.num_actions)
            return torch.tensor([action]).long()
        else:
            with torch.no_grad():
                return policy_net(state).argmax(dim=1).long()

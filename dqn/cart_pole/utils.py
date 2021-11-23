from collections import namedtuple
import os
import torch
import torchvision.transforms as t_transforms
import numpy as np
import gym
import matplotlib
import matplotlib.pyplot as plt

def get_moving_average(period, values):

    if len(values) >= period:
        values = torch.tensor(values, dtype=torch.float)

        moving_avg = values.unfold(
            dimension = 0,
            size      = period,
            step      = 1).mean(dim=1).flatten(start_dim = 0)

        moving_avg = torch.cat((torch.zeros(period - 1), moving_avg))
        return moving_avg.numpy()

    else:
        return np.zeros(len(values))


def plot(values, moving_avg_period, episode, epsilon):

    moving_avg = get_moving_average(moving_avg_period, values)

    #plt.figure(2)
    #plt.clf()
    #plt.title("Training")
    #plt.xlabel("Episode")
    #plt.ylabel("Duration")
    #plt.plot(values)
    #plt.plot(moving_avg)
    #plt.show()
    #plt.pause(0.001)
    print("\nEpsiode, epsilon: {}, {}".format(episode, epsilon))
    print("    Moving average: {}".format(moving_avg[-1]))


def extract_screen_tensors(experiences, device):

    batch = ScreenExperience(*zip(*experiences))

    state_t      = torch.cat(batch.state).to(device)
    action_t     = torch.cat(batch.action).to(device)
    reward_t     = torch.cat(batch.reward).to(device)
    next_state_t = torch.cat(batch.next_state).to(device)

    return (state_t, action_t, reward_t, next_state_t)

def extract_state_tensors(experiences, device):

    batch = StateExperience(*zip(*experiences))

    state_t      = torch.cat(batch.state).to(device)
    action_t     = torch.cat(batch.action).to(device)
    reward_t     = torch.cat(batch.reward).to(device)
    next_state_t = torch.cat(batch.next_state).to(device)
    done_t       = torch.cat(batch.done).to(device)

    return (state_t, action_t, reward_t, next_state_t, done_t)


ScreenExperience = namedtuple(
    'ScreenExperience',
    ('state', 'action', 'next_state', 'reward'))

StateExperience = namedtuple(
    'StateExperience',
    ('state', 'action', 'next_state', 'reward', 'done'))


class ReplayMemory(object):

    def __init__(self, capacity):

        self.capacity   = capacity
        self.memory     = np.array([None,] * capacity, dtype=object)
        self.push_count = 0

    def push(self, experience):

        self.memory[self.push_count % self.capacity] = experience
        self.push_count += 1

    def sample(self, batch_size):
        limit = min(self.push_count, self.capacity)
        idxs = np.arange(0, limit)
        idxs = np.random.choice(idxs, batch_size)
        return self.memory[idxs]

    def can_provide_sample(self, batch_size):
        return self.push_count >= batch_size

    def save(self, path):
        np.save(os.path.join(path, "replay_mem.npy"), self.memory, allow_pickle=True)

    def load(self, path):
        self.memory = np.load(os.path.join(path, "replay_mem.npy"), allow_pickle=True)


class EpsilonGreedyStrategy(object):

    def __init__(self, start, stop, decay):
        self.start = start
        self.stop  = stop
        self.decay = decay

    def get_exploration_rate(self, current_step):
        return self.stop + (self.start - self.stop) * \
            np.exp(-1. * current_step * self.decay)


class CartPoleEnvManager(object):

    def __init__(self, device):

        self.device         = device
        self.env            = gym.make("CartPole-v0").unwrapped
        self.current_screen = None
        self.done           = False

        self.env.reset()

    def reset(self):
        self.env.reset()
        self.current_screen = None

    def close(self):
        self.env.close()

    def render(self, mode='human'):
        return self.env.render(mode)

    def num_actions_available(self):
        return self.env.action_space.n

    def take_action(self, action):
        _, reward, self.done, _ = self.env.step(action.item())
        return torch.tensor([reward], device=self.device)

    def just_starting(self):
        return self.current_screen is None

    def get_screen_state(self):
        if self.just_starting() or self.done:
            self.current_screen = self.get_processed_screen()
            black_screen = torch.zeros_like(self.current_screen)
            return black_screen
        else:
            screen_1 = self.current_screen
            screen_2 = self.get_processed_screen()
            self.current_screen = screen_2
            return screen_2 - screen_1

    def get_screen_height(self):
        return self.get_processed_screen().shape[2]

    def get_screen_width(self):
        return self.get_processed_screen().shape[3]

    def get_processed_screen(self):
        screen = self.render("rgb_array").transpose((2, 0, 1))
        screen = self.crop_screen(screen)
        return self.transform_screen_data(screen)

    def crop_screen(self, screen):

        screen_height = screen.shape[1]
        top           = int(screen_height * 0.4)
        bottom        = int(screen_height * 0.8)

        screen_width  = screen.shape[2]
        left          = int(screen_width * 0.1)
        right         = int(screen_width * 0.9)

        screen        = screen[:, top : bottom, left : right]
        #screen        = screen[:, top : bottom, :]

        return screen

    def transform_screen_data(self, screen):

        screen = np.ascontiguousarray(screen, dtype=np.float32) / 255.
        screen = torch.from_numpy(screen)

        resize = t_transforms.Compose([
            t_transforms.ToPILImage(),
            t_transforms.Resize((40, 90)),
            t_transforms.ToTensor()])

        return resize(screen).unsqueeze(0).to(self.device)



class QValues(object):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @staticmethod
    def get_current(policy_net, states, actions):
        return policy_net(states).gather(dim=1, index=actions.unsqueeze(-1))

    @staticmethod
    def get_next_from_screen(target_net, next_states, next_actions):

        #
        # We need to figure out which states have not terminated, i.e. which
        # states did not end up in a black screen. These states need further
        # actions.
        #
        black_screen_locations = \
            next_states.flatten(start_dim = 1).max(dim = 1)\
            [0].eq(0).type(torch.bool)
 
        active_screen_locations = black_screen_locations == False
        active_states           = next_states[active_screen_locations]
        batch_size              = next_states.shape[0]
        values                  = torch.zeros(batch_size).to(QValues.device)
        next_actions            = next_actions[active_screen_locations]

        indices = torch.arange(active_states.shape[0]).long().to(QValues.device)

        values[active_screen_locations] = \
            target_net(active_states)[indices, next_actions].detach()

        return values

    @staticmethod
    def get_next_from_state(target_net, next_states, next_actions, dones):

        indices = torch.arange(next_states.shape[0]).long().to(QValues.device)
        next_q_values = target_net(next_states)[indices, next_actions]

        next_q_values[dones] = 0.0

        return next_q_values

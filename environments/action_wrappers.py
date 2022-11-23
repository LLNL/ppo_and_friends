from gym.spaces import MultiBinary
from abc import ABC, abstractmethod

class MultiBinaryWrapper(ABC):

    def __init__(self, env, **kw_args):
        self.env = env
        self.action_space      = self._get_action_space()
        self.observation_space = env.observation_space

    @abstractmethod
    def _get_action_space(self):
        return

    @abstractmethod
    def step(self, action):
        return

    def reset(self):
        return self.env.reset()

    def render(self, *args, **kw_args):
        self.env.render(*args, **kw_args)

    def seed(self, seed):
        self.env.seed(seed)

class MultiBinaryCartPoleWrapper(MultiBinaryWrapper):

    def _get_action_space(self):
        return MultiBinary(1)

    def step(self, action):
        return self.env.step(int(action.item()))


class MultiBinaryLunarLanderWrapper(MultiBinaryWrapper):

    def _get_action_space(self):
        return MultiBinary(2)

    def step(self, action):

        step_action = None

        action = action.flatten()

        if action.sum() == 0:
           step_action = 0
        elif action.sum() == 2:
            step_action = 1
        elif action[0] == 1 and action[1] == 0:
            step_action = 2
        elif action[0] == 0 and action[1] == 1:
            step_action = 3

        return self.env.step(step_action)

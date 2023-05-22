from ppo_and_friends.environments.action_wrappers import AlternateActionEnvWrapper
from gymnasium.spaces import MultiBinary

class MultiBinaryCartPoleWrapper(AlternateActionEnvWrapper):
    """
        A simple multi-binary action version of CartPole. This is for
        testing purposes only.
    """

    def _get_action_space(self):
        return MultiBinary(1)

    def step(self, action):
        return self.env.step(int(action.item()))

class MultiBinaryLunarLanderWrapper(AlternateActionEnvWrapper):
    """
        A simple multi-binary action version of LunarLander. This is for
        testing purposes only.
    """

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

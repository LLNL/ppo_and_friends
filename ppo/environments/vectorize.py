


class VectorizedEnv(object):

    def __init__(self,
                 env):

        self.env               = env
        self.observation_space = env.observation_space
        self.action_space      = env.action_space

    def step(self, action):
        obs, reward, done, info = self.env.step(action)

        if done:
            info["terminal obsveration"] = obs
            obs = self.env.reset()

        return obs, reward, done, info

    def reset(self):
        return self.env.reset()

    def render(self):
        self.env.render()

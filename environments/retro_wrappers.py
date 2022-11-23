
class RetroDeathWrapper():

    def __init__(self, env, **kw_args):
        self.env   = env
        self.lives = 0
        self.action_space      = env.action_space
        self.observation_space = env.observation_space

    def step(self, action):
        obs, reward, done, info = self.env.step(action)

        if self.lives > info["lives"]:
            reward -= 10.

        self.lives = info["lives"]

        return obs, reward, done, info

    def reset(self):
        self.lives = 0
        return self.env.reset()

    def render(self, *args, **kw_args):
        self.env.render(*args, **kw_args)

    def seed(self, seed):
        self.env.seed(seed)

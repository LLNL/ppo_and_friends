import torch
from utils.misc import get_action_type, need_action_squeeze
from environments.env_wrappers import ObservationNormalizer
import numpy as np

def test_policy(ppo,
                num_test_runs,
                device):

    env    = ppo.env
    policy = ppo.actor
    render = ppo.render

    action_type    = get_action_type(env)
    action_squeeze = need_action_squeeze(env)

    max_int     = np.iinfo(np.int32).max
    num_steps   = 0
    total_score = 0
    min_score   = max_int
    max_score   = -max_int

    policy.eval()

    for _ in range(num_test_runs):

        obs      = env.reset()
        done     = False
        ep_score = 0

        while not done:
            num_steps += 1

            if render:
                env.render()

            obs = torch.tensor(obs, dtype=torch.float).to(device).unsqueeze(0)

            with torch.no_grad():
                action = policy(obs).detach().cpu()

            if action_type == "continuous":
                action = torch.tanh(action)

            if action_type == "discrete":
                action = torch.argmax(action).numpy()
            else:
                action = action.numpy()

            if action_squeeze:
                action = action.squeeze()

            obs, reward, done, info = env.step(action)

            if "natural reward" in info:
                score = info["natural reward"]
            else:
                score = reward

            ep_score += score
            total_score += score

        min_score = min(min_score, ep_score)
        max_score = max(max_score, ep_score)

    print("Ran env {} times.".format(num_test_runs))
    print("Ran {} total time steps.".format(num_steps))
    print("Ran {} time steps on average.".format(num_steps / num_test_runs))
    print("Lowest score: {}".format(min_score))
    print("Highest score: {}".format(max_score))
    print("Average score: {}".format(total_score / num_test_runs))

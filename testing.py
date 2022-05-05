import torch
from ppo_and_friends.utils.misc import get_action_dtype
import numpy as np
from ppo_and_friends.utils.render import save_frames_as_gif
import os

def test_policy(ppo,
                render_gif,
                num_test_runs,
                device):
    """
        Test a trained policy.

        Arguments:
            ppo            An instance of PPO from ppo.py.
            render_gif     Create a gif from the renderings.
            num_test_runs  How many times should we run in the environment?
            device         The device to infer on.
    """

    env    = ppo.env
    policy = ppo.actor
    render = ppo.render

    action_dtype = get_action_dtype(env)

    max_int     = np.iinfo(np.int32).max
    num_steps   = 0
    total_score = 0
    min_score   = max_int
    max_score   = -max_int

    if render_gif:
        gif_frames = []

    policy.eval()

    for _ in range(num_test_runs):

        obs      = env.reset()
        done     = False
        ep_score = 0

        while not done:
            num_steps += 1

            if render:
                env.render()

            elif render_gif:
                gif_frames.append(env.render(mode = "rgb_array"))

            obs = torch.tensor(obs, dtype=torch.float).to(device).unsqueeze(0)

            with torch.no_grad():
                action = policy.get_result(obs).detach().cpu()

            if action_dtype == "discrete":
                action = torch.argmax(action, axis=-1).numpy()
            else:
                action = action.numpy()

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

    if render_gif:
        print("Attempting to create gif..")

        out_path = os.path.join(ppo.state_path, "GIF")
        if not os.path.exists(out_path):
            os.mkdir(out_path)

        save_frames_as_gif(
            frames   = gif_frames,
            out_path = out_path,
            filename = "test.gif")

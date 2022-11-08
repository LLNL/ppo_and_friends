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
    env        = ppo.env
    policies   = ppo.policies
    render     = ppo.render
    policy_map = ppo.policy_mapping_fn

    action_dtype = {}
    for agent_id in env.agent_ids:
        action_dtype[agent_id]= get_action_dtype(env.action_space[agent_id])

    max_int      = np.iinfo(np.int32).max
    num_steps    = 0
    total_scores = {agent_id : 0.0 for agent_id in env.agent_ids}
    min_scores   = {agent_id : max_int for agent_id in env.agent_ids}
    max_scores   = {agent_id : -max_int for agent_id in env.agent_ids}

    if render_gif:
        gif_frames = []

    for key in policies:
        policies[key].eval()

    for _ in range(num_test_runs):

        obs, _   = env.reset()
        done     = False

        while not done:
            num_steps += 1

            if render:
                env.render()

            elif render_gif:
                gif_frames.append(env.render(mode = "rgb_array"))

            actions = {}
            for agent_id in obs:

                obs[agent_id] = torch.tensor(obs[agent_id],
                    dtype=torch.float).to(device)

                obs[agent_id] = obs[agent_id].unsqueeze(0)
                policy_id     = policy_map(agent_id)

                with torch.no_grad():
                    actions[agent_id] = \
                        policies[policy_id].actor.get_result(
                            obs[agent_id]).detach().cpu()

                if action_dtype[agent_id] == "discrete":
                    actions[agent_id] = torch.argmax(actions[agent_id], axis=-1).numpy()
                else:
                    actions[agent_id] = actions[agent_id].numpy()

            obs, _, reward, done, info = env.step(actions)

            done = env.get_all_done()

            for agent_id in reward:
                reward[agent_id] = np.float32(reward[agent_id])

                if "natural reward" in info[agent_id]:
                    score = info[agent_id]["natural reward"]
                else:
                    score = reward[agent_id]

                total_scores[agent_id] += score

                min_scores[agent_id] = min(min_scores[agent_id], score)
                max_scores[agent_id] = max(max_scores[agent_id], score)

    for agent_id in env.agent_ids:
        print("\nAgent {}:".format(agent_id))
        print("    Ran env {} times.".format(num_test_runs))
        print("    Ran {} total time steps.".format(num_steps))
        print("    Ran {} time steps on average.".format(num_steps / num_test_runs))
        print("    Lowest score: {}".format(min_scores[agent_id]))
        print("    Highest score: {}".format(max_scores[agent_id]))
        print("    Average score: {}".format(total_scores[agent_id] / num_test_runs))

    if render_gif:
        print("Attempting to create gif..")

        out_path = os.path.join(ppo.state_path, "GIF")
        if not os.path.exists(out_path):
            os.mkdir(out_path)

        save_frames_as_gif(
            frames   = gif_frames,
            out_path = out_path,
            filename = "test.gif")

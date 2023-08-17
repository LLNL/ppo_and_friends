import torch
import sys
import dill as pickle
from ppo_and_friends.utils.misc import get_action_dtype
import numpy as np
from ppo_and_friends.utils.render import save_frames_as_gif
import os
from collections import OrderedDict

def test_policy(ppo,
                explore,
                num_test_runs,
                device,
                render_gif       = False,
                gif_fps          = 15,
                frame_pause      = 0.0,
                save_test_scores = False,
                **kw_args):
    """
        Test a trained policy.

        Arguments:
            ppo              An instance of PPO from ppo.py.
            explore          Bool determining whether or not exploration should
                             be enabled while testing.
            render_gif       Create a gif from the renderings.
            gif_fps          The frames per second for rendering a gif.
            num_test_runs    How many times should we run in the environment?
            device           The device to infer on.
            frame_pause      If rendering, sleep frame_pause seconds between
                             renderings.
            save_test_scores If True, save the agent scores to a pickled
                             dictionary.
    """
    env        = ppo.env
    policies   = ppo.policies
    render     = ppo.render

    action_dtype = OrderedDict({})
    for agent_id in env.agent_ids:
        action_dtype[agent_id]= get_action_dtype(env.action_space[agent_id])

    max_int      = np.iinfo(np.int32).max
    num_steps    = 0
    total_scores = OrderedDict({agent_id : 0.0 for agent_id in env.agent_ids})
    min_scores   = OrderedDict({agent_id : max_int for agent_id in env.agent_ids})
    max_scores   = OrderedDict({agent_id : -max_int for agent_id in env.agent_ids})

    if render_gif:
        gif_frames = []

    for key in policies:
        policies[key].eval()

    for _ in range(num_test_runs):

        obs, _   = ppo.apply_policy_reset_constraints(*env.reset())
        done     = False

        episode_score = OrderedDict({agent_id : 0.0 for agent_id in env.agent_ids})

        while not done:
            num_steps += 1

            if render:
                env.render(frame_pause = frame_pause)

            elif render_gif:
                gif_frames.append(env.render())

            actions = ppo.get_inference_actions(obs, explore)
            obs, _, reward, terminated, truncated, info = \
                ppo.apply_policy_step_constraints(*env.step(actions))

            done = env.get_all_done()

            for agent_id in reward:
                reward[agent_id] = np.float32(reward[agent_id])

                if "natural reward" in info[agent_id]:
                    score = info[agent_id]["natural reward"]
                else:
                    score = reward[agent_id]

                total_scores[agent_id]  += score
                episode_score[agent_id] += score

        for agent_id in total_scores:
            min_scores[agent_id] = min(min_scores[agent_id],
                episode_score[agent_id])
            max_scores[agent_id] = max(max_scores[agent_id],
                episode_score[agent_id])

    if save_test_scores:
        score_info = OrderedDict({})
        score_info["num_test_runs"]    = num_test_runs
        score_info["total_time_steps"] = num_steps

    for agent_id in env.agent_ids:
        print("\nAgent {}:".format(agent_id))
        print("    Ran env {} times.".format(num_test_runs))
        print("    Ran {} total time steps.".format(num_steps))
        print("    Ran {} time steps on average.".format(num_steps / num_test_runs))
        print("    Lowest score: {}".format(min_scores[agent_id]))
        print("    Highest score: {}".format(max_scores[agent_id]))
        print("    Average score: {}".format(total_scores[agent_id] / num_test_runs))

        if save_test_scores:
            score_info[agent_id] = {\
                "low_score"  : min_scores[agent_id],
                "high_score" : max_scores[agent_id],
                "avg_score"  : total_scores[agent_id] / num_test_runs,
            }

    if save_test_scores:
        score_file = os.path.join(ppo.state_path, "test-scores.pickle")

        with open(score_file, "wb") as out_f:
            pickle.dump(score_info, out_f,
                protocol=pickle.HIGHEST_PROTOCOL)

    if render_gif:
        print("Attempting to create gif..")

        out_path = os.path.join(ppo.state_path, "GIF")
        if not os.path.exists(out_path):
            os.mkdir(out_path)

        save_frames_as_gif(
            frames   = gif_frames,
            fps      = gif_fps,
            out_path = out_path,
            filename = "test.gif")

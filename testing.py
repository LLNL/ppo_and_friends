import torch
import sys
import yaml
import numpy as np
from ppo_and_friends.utils.render import save_frames_as_gif
import os

def test_policy(ppo,
                num_test_runs,
                deterministic    = False,
                render_gif       = False,
                gif_fps          = 15,
                frame_pause      = 0.0,
                save_test_scores = False,
                verbose          = False,
                **kw_args):
    """
    Test a trained policy.

    Paramters:
    ----------
    ppo: object
        An instance of PPO from ppo.py.
    deterministic: bool
        Bool determining whether or not we should sample our action
        prob distributions when testing.
    render_gif: bool
        Create a gif from the renderings.
    gif_fps: int
        The frames per second for rendering a gif.
    num_test_runs: int
        How many times should we run in the environment?
    frame_pause: float
        If rendering, sleep frame_pause seconds between renderings.
    save_test_scores: bool
        If True, save the agent scores to a yaml file.
    verbose: bool
        Enable verbosity?
    """
    env          = ppo.env
    policies     = ppo.policies
    render       = ppo.render
    max_int      = np.iinfo(np.int32).max
    num_steps    = 0
    total_policy_scores = {policy_id : 0.0 for policy_id in ppo.policies}
    min_policy_scores   = {policy_id : max_int for policy_id in ppo.policies}
    max_policy_scores   = {policy_id : -max_int for policy_id in ppo.policies}

    total_agent_scores  = {agent_id : 0.0 for agent_id in env.agent_ids}
    min_agent_scores    = {agent_id : max_int for agent_id in env.agent_ids}
    max_agent_scores    = {agent_id : -max_int for agent_id in env.agent_ids}

    if render_gif:
        gif_frames = []

    for key in policies:
        policies[key].eval()

    for _ in range(num_test_runs):

        obs, _   = ppo.apply_policy_reset_constraints(*env.reset())
        done     = False

        episode_agent_scores = \
            {agent_id : 0.0 for agent_id in env.agent_ids}

        episode_policy_scores = \
            {policy_id : 0.0 for policy_id in ppo.policies}

        while not done:
            num_steps += 1

            if render:
                env.render(frame_pause = frame_pause)

            elif render_gif:
                gif_frames.append(env.render())

            actions = ppo.get_inference_actions(obs, deterministic)
            obs, _, reward, terminated, truncated, info = \
                ppo.apply_policy_step_constraints(*env.step(actions))

            done = env.get_all_done()

            for agent_id in reward:
                reward[agent_id] = np.float32(reward[agent_id])

                if "natural reward" in info[agent_id]:
                    score = info[agent_id]["natural reward"]
                else:
                    score = reward[agent_id]

                policy_id = ppo.policy_mapping_fn(agent_id)
                total_policy_scores[policy_id]   += score
                episode_policy_scores[policy_id] += score

                total_agent_scores[agent_id]   += score
                episode_agent_scores[agent_id] += score

        for agent_id in total_agent_scores:
            min_agent_scores[agent_id] = min(min_agent_scores[agent_id],
                episode_agent_scores[agent_id])

            max_agent_scores[agent_id] = max(max_agent_scores[agent_id],
                episode_agent_scores[agent_id])

        for policy_id in total_policy_scores:
            min_policy_scores[policy_id] = min(min_policy_scores[policy_id],
                episode_policy_scores[policy_id])

            max_policy_scores[policy_id] = max(max_policy_scores[policy_id],
                episode_policy_scores[policy_id])

    if save_test_scores:
        score_info = {}
        score_info["num_test_runs"]    = num_test_runs
        score_info["total_time_steps"] = num_steps

    #
    # If verbose, print out scores for each agent. Otherwise, print out scores
    # for policies only.
    #
    if verbose:
        for agent_id in env.agent_ids:
            print("\nAgent {}:".format(agent_id))
            print("    Policy: {}".format(ppo.policy_mapping_fn(agent_id)))
            print("    Ran env {} times.".format(num_test_runs))
            print("    Ran {} total time steps.".format(num_steps))
            print("    Ran {} time steps on average.".format(num_steps / num_test_runs))
            print("    Lowest score: {}".format(min_agent_scores[agent_id]))
            print("    Highest score: {}".format(max_agent_scores[agent_id]))
            print("    Average score: {}".format(total_agent_scores[agent_id] / num_test_runs))
    else:
        for policy_id in ppo.policies:
            print("\nPolicy {}:".format(policy_id))
            print("    Ran env {} times.".format(num_test_runs))
            print("    Ran {} total time steps.".format(num_steps))
            print("    Ran {} time steps on average.".format(num_steps / num_test_runs))
            print("    Lowest score: {}".format(min_policy_scores[policy_id]))
            print("    Highest score: {}".format(max_policy_scores[policy_id]))
            print("    Average score: {}".format(total_policy_scores[policy_id] / num_test_runs))

    if save_test_scores:
        for agent_id in env.agent_ids:
            score_info[agent_id] = {\
                "low_score"  : float(min_agent_scores[agent_id]),
                "high_score" : float(max_agent_scores[agent_id]),
                "avg_score"  : float(total_agent_scores[agent_id] / num_test_runs),
                "policy"     : str(ppo.policy_mapping_fn(agent_id)),
            }

        for policy_id in ppo.policies:
            score_info[policy_id] = {\
                "low_score"  : float(min_policy_scores[policy_id]),
                "high_score" : float(max_policy_scores[policy_id]),
                "avg_score"  : float(total_policy_scores[policy_id] / num_test_runs),
            }

        score_file = os.path.join(ppo.state_path, "test-scores.yaml")

        with open(score_file, "w") as out_f:
            yaml.dump(score_info, out_f, default_flow_style=False)

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

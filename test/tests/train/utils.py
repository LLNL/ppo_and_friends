import sys
import dill as pickle
import subprocess
import os
from pathlib import Path

def get_root_dir():
    """
        Get the root directory for ppo_and_friends.

        Returns:
            The full path to the root directory of the ppo_and_friends
            repo.
    """
    test_path = os.path.realpath(__file__)
    root_dir  = Path(str(test_path).split("ppo_and_friends")[0])
    root_dir  = root_dir / "ppo_and_friends"
    return root_dir

def get_final_status(stdout):
    """
        Get the final status report from a string of training stdout.

        Arguments:
            stdout    A string capturing the stdout of a training.

        Returns:
            The final status report from the stdout.
    """
    try:
        final_status = stdout.split("Status Report:")[-1]
    except:
        final_status = None
    return final_status

def get_final_extrinsic_score_avg(stdout):
    """
        Get the final status report from a string of training stdout.

        Arguments:
            stdout    A string capturing the stdout of a training.

        Returns:
            The final status report from the stdout.
    """
    final_status = get_final_status(stdout)
    try:
        score = float(final_status.split("extrinsic score avg:")[1].split()[0])
    except:
        score = None
    return score

def extrinsic_score_training_test(name, train_command, passing_score):

    root_dir = get_root_dir()
    cur_dir  = Path(os.getcwd())

    train_command += f" --state-path {cur_dir}"
    print(f"Running command: {train_command}")

    os.chdir(root_dir)
    result = subprocess.run(train_command.split(),
        capture_output=True, text=True)
    os.chdir(cur_dir)

    final_status = get_final_status(result.stdout)
    score_avg    = get_final_extrinsic_score_avg(result.stdout)

    fail_msg  = f"\n************{name} FAILED************"
    fail_msg += f"\nExpected avg extrinsic score >= {passing_score}. "
    fail_msg += f"\nFinal score: {score_avg}"
    fail_msg += f"\nFinal status report: \n{final_status}"
    fail_msg += f"\nstderr: \n{result.stderr}"

    assert score_avg != None, fail_msg

    assert score_avg >= passing_score, fail_msg

    print(f"\n************{name} PASSED************")

def average_test_score_test(name, test_command, passing_scores, state_dir):

    root_dir = get_root_dir()
    cur_dir  = Path(os.getcwd())

    test_command += f" --state-path {cur_dir}"
    print(f"Running command: {test_command}")

    os.chdir(root_dir)
    result = subprocess.run(test_command.split(),
        capture_output=True, text=True)
    os.chdir(cur_dir)

    score_file = os.path.join("saved_states", state_dir, "test-scores.pickle")
    with open(score_file, "rb") as in_f:
        scores = pickle.load(in_f)

    fail_msg  = f"\n************{name} FAILED************"
    fail_msg += f"\nExpected avg scores:\n {passing_scores}"
    fail_msg += f"\nActual scores:\n {scores}"

    for agent_id in passing_scores:
        msg = f"ERROR: unable to find agent {agent_id} in the test scores."
        assert agent_id in scores, msg

        assert scores[agent_id]["avg_score"] >= passing_scores[agent_id], fail_msg

    print(f"\n************{name} PASSED************")


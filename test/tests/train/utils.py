import sys
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
    return stdout.split("Status Report:")[-1]

def get_final_extrinsic_score_avg(stdout):
    """
        Get the final status report from a string of training stdout.

        Arguments:
            stdout    A string capturing the stdout of a training.

        Returns:
            The final status report from the stdout.
    """
    final_status = get_final_status(stdout)
    return float(final_status.split("extrinsic score avg:")[1].split()[0])

def extrinsic_score_test(name, train_command):

    root_dir = get_root_dir()
    cur_dir  = Path(os.getcwd())

    os.chdir(root_dir)
    result = subprocess.run(train_command.split(),
        capture_output=True, text=True)
    os.chdir(cur_dir)

    final_status = get_final_status(result.stdout)
    score_avg    = get_final_extrinsic_score_avg(result.stdout)

    passing_score = 200.0

    fail_msg  = f"\n************{name} FAILED************"
    fail_msg += f"\nExpected avg extrinsic score >= {passing_score}. "
    fail_msg += f"\nFinal score: {score_avg}"
    fail_msg += f"\nFinal status report: \n{final_status}"
    fail_msg += f"\nstderr: \n{result.stderr}"

    assert score_avg >= passing_score, fail_msg

    print(f"\n************{name} PASSED************")

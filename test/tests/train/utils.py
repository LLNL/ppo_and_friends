import sys
import dill as pickle
import subprocess
import os
from pathlib import Path
import time
import shutil
import ppo_and_friends

def get_baseline_path():
    ppo_path = Path(ppo_and_friends.__file__).parent.absolute()
    return ppo_path / 'baselines'

def get_parallel_command():
    """
        Determine which command (if any) is available for
        distributed learning.

        Returns:
            A string containing the available command. None if there is
            no available command.
    """
    options = ['srun', 'mpirun']

    for cmd in options:
        if shutil.which(cmd) != None:
            return cmd
    return None

def run_command(command, verbose=False):
    print(f"Running command: {command}")
    t1 = time.time()
    result = subprocess.run(command.split(),
        capture_output=True, text=True)
    t2 = time.time()

    timing = (t2 - t1) / 60.
    print(f"Command finished in {timing} minutes.")

    if result.returncode != 0:
        print(result.stderr)

    if verbose:
        print(result.stdout)

    return result

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

def run_training(baseline_runner,
                 num_timesteps,
                 num_ranks     = 0,
                 options       = '',
                 verbose       = False):
    """
        Run training on a sub-process.

        Arguments:
            baseline_runner  The name of the baseline script to run.
                             This should be located in the baselines directory.
            num_timesteps    The number of timesteps to train for.
            num_ranks        The number of parallel ranks to train with.
            options (str)    Any other training options.
            verbose          Enable verbosity.
    """
    baseline_file  = os.path.join(get_baseline_path(), baseline_runner)
    train_command  = f"ppoaf --train {baseline_file} "
    train_command += f"--clobber --num-timesteps {num_timesteps} {options} "
    train_command += "--verbose "

    if num_ranks > 0:
        par_cmd = get_parallel_command()

        if par_cmd == None:
            msg  = "ERROR: unable to find a parallel library for distributed "
            msg += "training."
            sys.exit(1)

        train_command = f"{par_cmd} -n {num_ranks} {train_command} "

    cur_dir        = Path(os.getcwd())
    train_command += f"--state-path {cur_dir} "
    run_command(train_command, verbose)

def run_test(baseline_runner,
             num_test_runs,
             verbose = False):
    """
        Run a testing phase using a trained model.

        Arguments:
            baseline_runner  The name of the baseline script to run.
                             This should be located in the baselines directory.
            num_test_runs    The number of test runs to perform.
            verbose          Enable verbosity?
    """
    cur_dir       = Path(os.getcwd())
    baseline_file = os.path.join(get_baseline_path(), baseline_runner)
    test_command  = f"ppoaf --test {baseline_file} "
    test_command += f"--state-path {cur_dir} --save-test-scores "
    test_command += f"--num-test-runs {num_test_runs} --verbose "

    run_command(test_command, verbose)

def average_score_test(name,
                       baseline_runner,
                       num_test_runs,
                       passing_scores,
                       verbose = False):
    """
        Run a testing phase using a trained model and determine if
        the model reaches passing average scores.

        Arguments:
            name             The name of the test.
            baseline_runner  The name of the baseline script to run.
                             This should be located in the baselines directory.
            num_test_runs    The number of test runs to perform.
            passing_scores   A dict containing passing scores for each
                             agent.
            verbose          Enable verbosity?
    """
    run_test(baseline_runner, num_test_runs)

    cur_dir    = Path(os.getcwd())
    state_dir  = os.path.basename(baseline_runner).split('.')[:-1][0]
    score_file = os.path.join(cur_dir, "saved_states", state_dir,
        "test-scores.pickle")

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

def high_score_test(name,
                    baseline_runner,
                    num_test_runs,
                    passing_scores,
                    verbose = False):
    """
        Run a testing phase using a trained model and determine if
        the model reaches passing high scores.

        Arguments:
            name             The name of the test.
            baseline_runner  The name of the baseline script to run.
                             This should be located in the baselines directory.
            num_test_runs    The number of test runs to perform.
            passing_scores   A dict containing passing scores for each
                             agent.
            verbose          Enable verbosity?
    """
    run_test(baseline_runner, num_test_runs)

    cur_dir    = Path(os.getcwd())
    state_dir  = os.path.basename(baseline_runner).split('.')[:-1][0]
    score_file = os.path.join(cur_dir, "saved_states", state_dir,
        "test-scores.pickle")

    with open(score_file, "rb") as in_f:
        scores = pickle.load(in_f)

    fail_msg  = f"\n************{name} FAILED************"
    fail_msg += f"\nExpected high scores:\n {passing_scores}"
    fail_msg += f"\nActual scores:\n {scores}"

    for agent_id in passing_scores:
        msg = f"ERROR: unable to find agent {agent_id} in the test scores."
        assert agent_id in scores, msg

        assert scores[agent_id]["high_score"] >= passing_scores[agent_id], fail_msg

    print(f"\n************{name} PASSED************")


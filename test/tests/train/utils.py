import sys
import yaml
import subprocess
import os
from pathlib import Path
import time
import shutil
import ppo_and_friends

def get_baseline_path():
    ppo_path = Path(ppo_and_friends.__file__).parent.absolute()
    return ppo_path / 'baselines'

def get_state_path(baseline_runner):
    cur_dir       = Path(os.getcwd())
    runner_name   = os.path.basename(baseline_runner).split('.')[:-1][0]
    state_path    = os.path.join(cur_dir, "saved_states", runner_name)
    return state_path

def get_parallel_command():
    """
    Determine which command (if any) is available for
    distributed learning.

    Returns:
    --------
    str:
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

    Parameters:
    -----------
    stdout: str
        A string capturing the stdout of a training.

    Returns:
    --------
    str:
        The final status report from the stdout.
    """
    try:
        final_status = stdout.split("Status Report:")[-1]
    except:
        final_status = None
    return final_status

def run_training(baseline_type,
                 baseline_runner,
                 num_timesteps,
                 num_ranks     = 0,
                 options       = '',
                 verbose       = False,
                 random_seed   = 1):
    """
    Run training on a sub-process.

    Parameters:
    -----------
    baseline_type: str
        What type of baseline is this?
    baseline_runner: str
        The name of the baseline script to run.
        This should be located in the baselines directory.
    num_timesteps: int
        The number of timesteps to train for.
    num_ranks: int
        The number of parallel ranks to train with.
    options: str
        Any other training options.
    verbose: bool
        Enable verbosity.
    random_seed: int
        The random seed to use.
    """
    baseline_file  = os.path.join(get_baseline_path(), baseline_type, baseline_runner)
    train_command  = f"ppoaf train {baseline_file} "
    train_command += f"--clobber --num_timesteps {num_timesteps} {options} "
    train_command += f"--verbose --random_seed {random_seed} "

    if num_ranks > 0:
        par_cmd = get_parallel_command()

        if par_cmd == None:
            msg  = "ERROR: unable to find a parallel library for distributed "
            msg += "training."
            sys.exit(1)

        train_command = f"{par_cmd} -n {num_ranks} {train_command} "

    cur_dir        = Path(os.getcwd())
    train_command += f"--state_path {cur_dir}/saved_states "
    run_command(train_command, verbose)

def run_test(baseline_runner,
             num_test_runs,
             deterministic = False,
             verbose       = False,
             random_seed   = 1,
             options       = ""):
    """
    Run a testing phase using a trained model.

    Parameters:
    -----------
    baseline_runner: str
        The name of the baseline script to run.
        This should be located in the baselines directory.
    num_test_runs: int
        The number of test runs to perform.
    deterministic: bool
        Enable determinism while testing?
    verbose: bool
        Enable verbosity?
    random_seed: int
        The random seed to use.
    options: str
        Any other training options.
    """
    output_dir    = get_state_path(baseline_runner)
    test_command  = f"ppoaf test {output_dir} "
    test_command += f"--save_test_scores "
    test_command += f"--num_test_runs {num_test_runs} --verbose --random_seed {random_seed} {options} "

    if deterministic:
        test_command += f"--deterministic"

    run_command(test_command, verbose)

def average_score_test(name,
                       baseline_runner,
                       num_test_runs,
                       passing_scores,
                       deterministic = False,
                       verbose       = False,
                       **kw_args):
    """
    Run a testing phase using a trained model and determine if
    the model reaches passing average scores.

    Parameters:
    -----------
    name: str
        The name of the test.
    baseline_runner: str
        The name of the baseline script to run.
        This should be located in the baselines directory.
    num_test_runs: int
        The number of test runs to perform.
    passing_scores: dict
        A dict containing passing scores for each agent.
    deterministic: bool
        Enable deterministic while testing?
    verbose: bool
        Enable verbosity?
    """
    run_test(baseline_runner, num_test_runs, deterministic, **kw_args)

    state_path = get_state_path(baseline_runner)
    score_file = os.path.join(state_path,
        "test-scores.yaml")

    with open(score_file, "r") as in_f:
        scores = yaml.safe_load(in_f)

    fail_msg  = f"\n************{name} FAILED************"
    fail_msg += f"\nExpected vs actual avg scores:\n "
    for agent_id in passing_scores:
        fail_msg += f"    {agent_id}: {passing_scores[agent_id]} vs {scores[agent_id]['avg_score']}\n"

    for agent_id in passing_scores:
        msg = f"ERROR: unable to find agent {agent_id} in the test scores."
        assert agent_id in scores, msg

        assert scores[agent_id]["avg_score"] >= passing_scores[agent_id], fail_msg

    print(f"\n************{name} PASSED************")

def high_score_test(name,
                    baseline_runner,
                    num_test_runs,
                    passing_scores,
                    deterministic = False,
                    verbose       = False,
                    **kw_args):
    """
    Run a testing phase using a trained model and determine if
    the model reaches passing high scores.

    Parameters:
    -----------
    name: str
        The name of the test.
    baseline_runner: str
        The name of the baseline script to run.
        This should be located in the baselines directory.
    num_test_runs: str
        The number of test runs to perform.
    passing_scores: dict
        A dict containing passing scores for each agent.
    deterministic: bool
        Enable deterministic while testing?
    verbose: bool
        Enable verbosity?
    """
    run_test(baseline_runner, num_test_runs, deterministic, **kw_args)

    state_path = get_state_path(baseline_runner)
    score_file = os.path.join(state_path,
        "test-scores.yaml")

    with open(score_file, "r") as in_f:
        scores = yaml.safe_load(in_f)

    fail_msg  = f"\n************{name} FAILED************"
    fail_msg += f"\nExpected vs actual high scores:\n "
    for agent_id in passing_scores:
        fail_msg += f"    {agent_id}: {passing_scores[agent_id]} vs {scores[agent_id]['high_score']}\n"

    for agent_id in passing_scores:
        msg = f"ERROR: unable to find agent {agent_id} in the test scores."
        assert agent_id in scores, msg

        assert scores[agent_id]["high_score"] >= passing_scores[agent_id], fail_msg

    print(f"\n************{name} PASSED************")


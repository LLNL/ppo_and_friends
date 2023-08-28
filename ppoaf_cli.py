import torch
import random
import numpy as np
import argparse
import sys
import importlib.util
import inspect
import os
import re
from ppo_and_friends.utils.mpi_utils import rank_print
import shutil
from mpi4py import MPI

comm      = MPI.COMM_WORLD
rank      = comm.Get_rank()
num_procs = comm.Get_size()


def cli():

    parser = argparse.ArgumentParser()

    parser.add_argument("--train", type=str, default='',
        help="A path to a file containing the environment runner to train.")

    parser.add_argument("--test", type=str, default='',
        help="A path to a file containing the environment runner to test.")

    parser.add_argument("--test-explore", action="store_true",
        help="Enable exploration while testing. Note that this flag"
        "only has an effect while in test mode. Exploration is always"
        "enabled during training.")

    parser.add_argument("--num-test-runs", type=int, default=1,
        help="If used with --test, this will define the number of test "
        "iterations that are run. The min, max, and average scores will "
        "be reported.")

    parser.add_argument("--save-test-scores", action="store_true",
        help="If used with --test, the test scores for each agent will be "
        "saved as a pickle file in the output directory.")

    parser.add_argument("--save-train-scores", action="store_true",
        help="If True, each policy's extrinsic score average will be saved "
        "in a text file every iteration. Each agent will have an individual "
        "file in the state directory.")

    parser.add_argument("--device", type=str, default="cpu",
        help="Which device to use for training.")

    parser.add_argument("--state-path", default="./",
        help="Where to save states and policy info. The data will be "
        "saved in a sub-directory named 'saved_states'.")

    parser.add_argument("--clobber", action="store_true",
        help="Clobber any existing saves associated with this environment.")

    parser.add_argument("--render", action="store_true",
        help="Render the environment at each step.")

    parser.add_argument("--render-gif", action="store_true",
        help="Render a gif when testing.")

    parser.add_argument("--gif-fps", type=int, default=15,
        help="The frames per second for rendering a gif.")

    parser.add_argument("--frame-pause", default=0.0, type=float,
        help="When rendering, pause between frames for frame-pause seconds."
        "Note that this flag only works when used in conjunction with the "
        "--render flag.")

    parser.add_argument("--verbose", action="store_true",
        help="Enable verbosity.")

    parser.add_argument("--pickle-class", action="store_true",
        help="Pickle the entire PPO class. If True, the pickled class will be "
        "saved in the state-path. This is useful for loading a trained model "
        "for inference outside of this workflow.")

    #TODO: let's also let users stop at an iteration rather than timestep.
    parser.add_argument("--num-timesteps", default=1000000, type=int,
        help="The number of timesteps to train for.")

    parser.add_argument("--random-seed", default=0, type=int,
        help="The random seed to use.")

    parser.add_argument("--envs-per-proc", default=1, type=int,
        help="The number of environment instances each processor should have.")

    parser.add_argument("--force-deterministic", action="store_true",
        help="Tell PyTorch to only use deterministic algorithms.")

    args               = parser.parse_args()
    train              = args.train != ''
    test               = args.test != ''
    test_explore       = args.test_explore
    random_seed        = args.random_seed + rank
    num_test_runs      = args.num_test_runs
    save_test_scores   = args.save_test_scores
    clobber            = args.clobber
    render             = args.render
    render_gif         = args.render_gif
    gif_fps            = args.gif_fps
    frame_pause        = args.frame_pause
    verbose            = args.verbose
    num_timesteps      = args.num_timesteps
    force_determinism  = args.force_deterministic
    envs_per_proc      = args.envs_per_proc
    pickle_class       = args.pickle_class
    device             = torch.device(args.device)
    save_train_scores  = args.save_train_scores
    runner_file        = args.train if args.train != '' else args.test

    if (not train) and (not test):
        msg = "ERROR: args train or test must be specified."
        rank_print(msg)
        comm.Abort()

    if train and test:
        msg = "ERROR: training and testing cannot be done simultaneously."
        rank_print(msg)
        comm.Abort()

    if render and render_gif:
        msg  = "ERROR: render and render_gif are both enabled, "
        msg += "but they cannot be used simultaneously."
        rank_print(msg)
        comm.Abort()

    if render_gif and not test:
        msg = "ERROR: render_gif is only allowed when testing."
        rank_print(msg)
        comm.Abort()

    if num_test_runs > 1 and not test:
        msg = "ERROR: --num_test_runs can only be used with --test."
        rank_print(msg)
        comm.Abort()

    #
    # Set random seeds (this doesn't guarantee reproducibility, but it should
    # help).
    #
    if random_seed >= 0:
        torch.manual_seed(random_seed)
        random.seed(random_seed)
        np.random.seed(random_seed)

    if force_determinism:
        torch.use_deterministic_algorithms(True)

    spec = importlib.util.spec_from_file_location("EnvRunner", runner_file)
    env_runner  = importlib.util.module_from_spec(spec)
    
    sys.modules["EnvRunner"] = env_runner
    spec.loader.exec_module(env_runner)
    
    valid_runners = []
    valid_names   = []
    for name, obj in inspect.getmembers(env_runner):
        if inspect.isclass(obj):
            if '_ppoaf_runner_tag' in dir(obj):
                valid_runners.append(obj)
                valid_names.append(name)

    if len(valid_runners) > 1:
        msg  = "ERROR: found more than one environment runner in "
        msg += f"the runner file: {valid_names}"
        rank_print(msg)
        comm.Abort()

    if len(valid_runners) == 0:
        msg  = "ERROR: unable to find a valid runner in the given file. "
        msg += "Make sure that you've added the correct decorator to your "
        msg += "runner."
        rank_print(msg)
        comm.Abort()

    file_preface = os.path.basename(runner_file).split('.')[:-1][0]
    state_path   = os.path.join(args.state_path, "saved_states", file_preface)
    runner_class = valid_runners[0]

    load_state = not clobber or test

    if clobber and rank == 0:
        if os.path.exists(state_path):
            shutil.rmtree(state_path)
    comm.barrier()

    rank_print("Using device: {}".format(device))
    rank_print("Number of processors: {}".format(num_procs))
    rank_print("Number of environments per processor: {}".format(envs_per_proc))

    runner = runner_class(
        random_seed           = random_seed,
        state_path            = state_path,
        load_state            = load_state,
        render                = render,
        render_gif            = render_gif,
        gif_fps               = gif_fps,
        frame_pause           = frame_pause,
        verbose               = verbose,
        num_timesteps         = num_timesteps,
        device                = device,
        envs_per_proc         = envs_per_proc,
        test                  = test,
        explore_while_testing = test_explore,
        save_test_scores      = save_test_scores,
        save_train_scores     = save_train_scores,
        pickle_class          = pickle_class,
        num_test_runs         = num_test_runs)

    runner.run()

if __name__ == "__main__":
    cli()

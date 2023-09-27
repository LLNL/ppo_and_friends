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
from ppo_and_friends.utils.plot_scores import plot_score_files
import shutil
from mpi4py import MPI

comm      = MPI.COMM_WORLD
rank      = comm.Get_rank()
num_procs = comm.Get_size()


def cli():

    parent_parser = argparse.ArgumentParser(add_help=False)

    parent_parser.add_argument("--device", type=str, default="cpu",
        help="Which device to use for training.")

    parent_parser.add_argument("--state-path", default="./saved_states",
        help="Where to save states and policy info. The data will be "
        "saved in a sub-directory named 'saved_states'.")

    parent_parser.add_argument("--render", action="store_true",
        help="Render the environment at each step.")

    parent_parser.add_argument("--frame-pause", default=0.0, type=float,
        help="When rendering, pause between frames for frame-pause seconds."
        "Note that this flag only works when used in conjunction with the "
        "--render flag.")

    parent_parser.add_argument("--verbose", action="store_true",
        help="Enable verbosity.")

    parent_parser.add_argument("--pickle-class", action="store_true",
        help="Pickle the entire PPO class. If True, the pickled class will be "
        "saved in the state-path. This is useful for loading a trained model "
        "for inference outside of this workflow.")

    parent_parser.add_argument("--random-seed", default=0, type=int,
        help="The random seed to use.")

    parent_parser.add_argument("--force-deterministic", action="store_true",
        help="Tell PyTorch to only use deterministic algorithms.")

    parent_parser.add_argument("--force-gc", action="store_true",
        help="Force manually garbage collection. This will slow down "
        "computations, but it can alleviate memory bottlenecks.")

    parent_parser.add_argument("--state-tag", type=str, default="",
        help="An optional string representing a unique 'tag' for a given "
        "run. This tag will be appended to the output directory so that "
        "the saved state is located in <state-path>/<runner-name>-<tag>. "
        "This allows for multiple runs of the same runner with different "
        "parameters to be saved to the same state path while remaining "
        "distinct.")

    main_parser = argparse.ArgumentParser()

    #
    # Create a subparser for the different commands.
    #
    subparser = main_parser.add_subparsers(dest='command', required=True)

    #
    # 'train' command subparser
    #
    train_parser = subparser.add_parser("train",
        help="Train using a PPO-AF runner.", parents=[parent_parser])

    train_parser.add_argument("runner", type=str,
        help="Path to the runner to train.")

    train_parser.add_argument("--save-train-scores", action="store_true",
        help="If True, each policy's extrinsic score average will be saved "
        "in a text file every iteration. Each agent will have an individual "
        "file in the state directory.")

    train_parser.add_argument("--clobber", action="store_true",
        help="Clobber any existing saves associated with this environment.")

    #TODO: let's also let users stop at an iteration rather than timestep.
    train_parser.add_argument("--num-timesteps", default=1000000, type=int,
        help="The number of timesteps to train for.")

    train_parser.add_argument("--envs-per-proc", default=1, type=int,
        help="The number of environment instances each processor should have.")

    #
    # 'test' command subparser
    #
    test_parser = subparser.add_parser("test",
        help="Evaluate a trained PPO-AF runner.", parents=[parent_parser])

    test_parser.add_argument("runner", type=str,
        help="Path to the runner to evaluate.")

    test_parser.add_argument("--test-explore", action="store_true",
        help="Enable exploration while testing. Note that this flag"
        "only has an effect while in test mode. Exploration is always"
        "enabled during training.")

    test_parser.add_argument("--num-test-runs", type=int, default=1,
        help="If used with --test, this will define the number of test "
        "iterations that are run. The min, max, and average scores will "
        "be reported.")

    test_parser.add_argument("--save-test-scores", action="store_true",
        help="If used with --test, the test scores for each agent will be "
        "saved as a pickle file in the output directory.")

    test_parser.add_argument("--render-gif", action="store_true",
        help="Render a gif when testing.")

    test_parser.add_argument("--gif-fps", type=int, default=15,
        help="The frames per second for rendering a gif.")

    #
    # 'plot' command subparser
    #
    plot_parser = subparser.add_parser("plot",
        help="Plot reward curves from trained policies.", parents=[parent_parser])

    plot_parser.add_argument("scores", type=str, nargs="+", help="Paths to the "
        "policy score files that you wish to plot. This can be paths "
        "to the actual score files, directories containing the score files, "
        "or directories containing sub-directories (at any depth) containing "
        "score files.")

    args, unknown_args = main_parser.parse_known_args()

    if args.command == "plot":
        plot_score_files(args.scores)
        return

    if len(unknown_args) > 0:
        msg  = f"WARNING: the following args are unrecognized by the "
        msg += f"primary PPOAF parser. Ignore this message if they are "
        msg += f"used by the runner: {unknown_args}"
        rank_print(msg)

    arg_dict = vars(args)
    arg_dict["device"]      = torch.device(args.device)
    arg_dict["random_seed"] = arg_dict["random_seed"] + rank

    random_seed             = arg_dict["random_seed"]
    runner_file             = arg_dict["runner"]
    force_deterministic     = arg_dict["force_deterministic"]

    if args.command == 'test':
        if args.render and args.render_gif:
            msg  = "ERROR: render and render_gif are both enabled, "
            msg += "but they cannot be used simultaneously."
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

    if force_deterministic:
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
    arg_dict["state_path"] = os.path.join(
        args.state_path , file_preface)

    if args.state_tag != "":
        arg_dict["state_path"] = f"{arg_dict['state_path']}-{args.state_tag}"

    runner_class = valid_runners[0]

    rank_print("Using device: {}".format(args.device))
    rank_print("Number of processors: {}".format(num_procs))

    if args.command == "train":
        clobber    = arg_dict["clobber"]
        load_state = False

        if clobber:
            if rank == 0:
                if os.path.exists(arg_dict["state_path"]):
                    shutil.rmtree(arg_dict["state_path"])

        elif os.path.exists(os.path.join(arg_dict["state_path"], "state_0.pickle")):
            load_state = True

        comm.barrier()

        envs_per_proc = arg_dict["envs_per_proc"]
        rank_print("Number of environments per processor: {}".format(envs_per_proc))

    elif args.command == "test":
        if not  os.path.exists(os.path.join(arg_dict["state_path"], "state_0.pickle")):
            rank_print("ERROR: unable to find saved state for testing!")
            comm.Abort()

        load_state = True

    runner = runner_class(
        **arg_dict,
        test       = args.command == "test",
        load_state = load_state)

    #
    # Allow the runner to add more args to the parser if needed and
    # store them internally.
    #
    parent_parser = runner.parse_extended_cli_args(parent_parser)

    runner.run()

if __name__ == "__main__":
    cli()

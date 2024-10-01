import torch
import shutil
import random
import numpy as np
import argparse
import yaml
import sys
import dill as pickle
import importlib.util
import inspect
import os
import re
import ast
from ppo_and_friends.utils.mpi_utils import rank_print
from ppo_and_friends.utils.plotting import plot_curve_files
import shutil
from mpi4py import MPI

comm      = MPI.COMM_WORLD
rank      = comm.Get_rank()
num_procs = comm.Get_size()


def update_pretrained(arg_dict):
    """
    Validate and update the pretrained_policies argument in the
    parsed args dict.

    Parameters:
    -----------
    arg_dict: dict
        The argument dictionary from parsed args.
    """
    pretrained_policies = arg_dict["pretrained_policies"]

    if len(pretrained_policies) == 0:
        msg  = "ERROR: invalid pretrained policies given, "
        msg += "{pretrained_policies}"
        rank_print(msg)
        comm.Abort()

    #
    # If the string looks like a dictionary, we turn it into one. Otherwise,
    # it better be an existing state path.
    #
    if pretrained_policies[0] == "{":
        pretrained_policies = ast.literal_eval(pretrained_policies)

    elif not os.path.exists(pretrained_policies):
        msg  = "ERROR: pretrained_policies path {pretrained_policies} "
        msg += "does not exist."
        rank_print(msg)
        comm.Abort()

    arg_dict["pretrained_policies"] = pretrained_policies
    return arg_dict

def get_runner_class(runner_file):
    """
    Get the runner class from a runner file.

    Parameters:
    -----------
    runner_file: str
        A path to a runner file.

    Returns:
    --------
    class:
        A runner class.
    """
    spec = importlib.util.spec_from_file_location("EnvRunner", runner_file)
    env_runner = importlib.util.module_from_spec(spec)

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

    return valid_runners[0]


def cli():

    parent_parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)

    parent_parser.add_argument("--device", type=str, default="cpu",
        help="Which device to use for training.")

    parent_parser.add_argument("--render", action="store_true",
        help="Render the environment at each step.")

    parent_parser.add_argument("--frame_pause", default=0.0, type=float,
        help="When rendering, pause between frames for frame_pause seconds."
        "Note that this flag only works when used in conjunction with the "
        "--render flag.")

    parent_parser.add_argument("--verbose", action="store_true",
        help="Enable verbosity.")

    parent_parser.add_argument("--pickle_class", action="store_true",
        help="Pickle the entire PPO class. If True, the pickled class will be "
        "saved in the state-path. This is useful for loading a trained model "
        "for inference outside of this workflow.")

    parent_parser.add_argument("--random_seed", default=0, type=int,
        help="The random seed to use.")

    parent_parser.add_argument("--force_deterministic", action="store_true",
        help="Tell PyTorch to only use deterministic algorithms.")

    parent_parser.add_argument("--force_gc", action="store_true",
        help="Force manually garbage collection. This will slow down "
        "computations, but it can alleviate memory bottlenecks.")

    parent_parser.add_argument("--state_tag", type=str, default="",
        help="An optional string representing a unique 'tag' for a given "
        "run. This tag will be appended to the output directory so that "
        "the saved state is located in <state_path>/<runner_name>-<tag>. "
        "This allows for multiple runs of the same runner with different "
        "parameters to be saved to the same state path while remaining "
        "distinct.")

    parent_parser.add_argument("--policy_tag", type=str, default="latest",
        help="An optional string representing a unique 'tag' used when "
        "loading previously saved policies. Common options are 'latest', "
        "'<policy_name>_best', and '<checkpoint_iteration>'. Note that this "
        "argument is ignored when pretrained_policies is used.")

    main_parser = argparse.ArgumentParser(allow_abbrev=False)

    #
    # Create a subparser for the different commands.
    #
    subparser = main_parser.add_subparsers(dest='command', required=True)

    #
    # 'train' command subparser
    #
    train_parser = subparser.add_parser("train", allow_abbrev=False,
        help="Train using a PPO-AF runner.", parents=[parent_parser])

    train_parser.add_argument("runner", type=str,
        help="Path to the runner to train.")

    train_parser.add_argument("--state_path", default="./saved_states",
        help="Where to save states and policy info.")

    train_parser.add_argument("--disable_save_train_scores",
        action="store_true", help="By default, the 'natural score avg' for "
        "each agent is saved into a numpy file during training. Enabling this "
        "flag will disable this feature, and the scores will not be saved.")

    train_parser.add_argument("--disable_save_avg_ep_len",
        action="store_true", help="By default, the average episode length "
        "is saved into a numpy file during training. Enabling this "
        "flag will disable this feature.")

    train_parser.add_argument("--disable_save_running_time",
        action="store_true", help="By default, the running time "
        "is saved into a numpy file during training. Enabling this "
        "flag will disable this feature.")

    train_parser.add_argument("--clobber", action="store_true",
        help="Clobber any existing saves associated with this environment.")

    train_parser.add_argument("--num_timesteps", default=1000000, type=int,
        help="The number of timesteps to train for.")

    train_parser.add_argument("--envs_per_proc", default=1, type=int,
        help="The number of environment instances each processor should have.")

    train_parser.add_argument("--pretrained_policies", default="{}", type=str,
        help="Where to load pre-trained policies from. This can either be a "
        "string to a single state path where all policies should be loaded "
        "from (latest saves) or a dictionary mapping policy ids to specific "
        "save directories. Dict example: {'policy_a' : '/foo/my-game/adversary-policy/latest', "
        "'policy_b' : '/foo/my-game-2/agent-policy/100'}. String example: "
        "'/foo/my-game/'.")

    train_parser.add_argument("--env_state", default=None, type=str,
        help="An optional path to load pre-trained environment state from. "
        "This will include normalizations like observation normalizers and "
        "useful when loading pre-trained policies.")

    train_parser.add_argument("--freeze_policies", nargs="+", type=str,
        help="Which policies to 'freeze' the weights of. These policies "
        "will NOT be further trained; they merely act in the environment.")

    train_parser.add_argument("--checkpoint_every", default=20, type=int,
        help="How often to save checkpoints in relation to iterations.")

    #
    # 'test' command subparser
    #
    test_parser = subparser.add_parser("test", allow_abbrev=False,
        help="Evaluate a trained PPO-AF runner.", parents=[parent_parser])

    test_parser.add_argument("state_path", type=str,
        help="Path to the saved state to evaluate.")

    test_parser.add_argument("--deterministic", action="store_true",
        help="Make the actor behave deterministically. If used, "
        "the action probabilities will not be sampled. Instead, we "
        "will always act using the highest probability action.")

    test_parser.add_argument("--num_test_runs", type=int, default=1,
        help="This will define the number of test "
        "iterations that are run. The min, max, and average scores will "
        "be reported.")

    test_parser.add_argument("--save_test_scores", action="store_true",
        help="The test scores for each agent will be "
        "saved to a yaml file in the output directory.")

    test_parser.add_argument("--render_gif", action="store_true",
        help="Render a gif when testing.")

    test_parser.add_argument("--gif_fps", type=int, default=15,
        help="The frames per second for rendering a gif.")

    test_parser.add_argument("--pretrained_policies", default="{}", type=str,
        help="Where to load pre-trained policies from. This can either be a "
        "string to a single state path where all policies should be loaded "
        "from (latest saves) or a dictionary mapping policy ids to specific "
        "save directories. Dict example: {'policy_a' : '/foo/my-game/adversary-policy/latest', "
        "'policy_b' : '/foo/my-game-2/agent-policy/100'}. String example: "
        "'/foo/my-game/'.")

    test_parser.add_argument("--env_state", default=None, type=str,
        help="An optional path to load pre-trained environment state from. "
        "This will include normalizations like observation normalizers and "
        "useful when loading pre-trained policies.")

    #
    # 'plot' command subparser
    #
    plot_parser = subparser.add_parser("plot", allow_abbrev=False,
        help="Plot reward curves from trained policies.")

    plot_parser.add_argument("curves", type=str, nargs="+", help="Paths to the "
        "policy score files that you wish to plot. This can be paths "
        "to the actual score files, directories containing the score files, "
        "or directories containing sub-directories (at any depth) containing "
        "score files.")

    plot_parser.add_argument("--curve_type", type=str, default="scores",
        choices=["scores", "runtime", "episode_length",
        "bs_min", "bs_max", "bs_avg", "episode_scores"],
        help="The 'curve_type' is used to refine searches for saved curves "
        "when the curve file is not explicitly set. For instance, if "
        "a user sets 'curves' to a directory to be searched, the searching "
        "algorithm will look for subdirectories named <curve_type> within "
        "the 'curves' directory of state paths. Only these curves will be "
        "collected.")

    plot_parser.add_argument("--save_path", type=str, default="",
        help="Optional path to save a figure to instead of rendering in a "
        "window. The the file should have an extension that is supported "
        "by plotly.")

    plot_parser.add_argument("--title", type=str, default="",
        help="The plot title to use.")

    plot_parser.add_argument("--inclusive_search_patterns", "--isp",
        type=str, nargs="+",
        help="Only grab plot files that contain these strings within "
        "their path. ALL strings must be contained in every path.")

    plot_parser.add_argument("--exclusive_search_patterns", "--esp",
        type=str, nargs="+",
        help="Only grab plot files that contain these strings within "
        "their path. Only ONE string must be contained in each path.")

    plot_parser.add_argument("--exclude_patterns", type=str, nargs="+",
        help="Exclude any plot files that contain these strings in their "
        "path.")

    plot_parser.add_argument("--status_constraints", type=ast.literal_eval,
        default="{}",
        help="A dictionary of status constraints passed as a string. The "
        "format is {'status_name_0' : ('comp_func_0', comp_val_0), 'status_preface' "
        ": {'status_name_1' : ('comp_func_1', comp_val_1)}} s.t. 'comp_func_i' is "
        "one of <, >, <=, >=, =.")

    plot_parser.add_argument("--grouping", action="store_true",
        help="If enabled, all curves from specified paths will be "
        "grouped together by their given paths. The curves within "
        "each group will be used to define a deviation and mean, which will be "
        "plotted per group.")

    plot_parser.add_argument("--group_names", type=str, nargs="+",
        help="If --grouping is True, this flag can be used to assign "
        "names to each of the groups. The number of group_names MUST "
        "match the number of groups.")

    plot_parser.add_argument("--group_deviation", type=str, default="std",
        choices=["std", "min_max"],
        help="If --grouping is True, this flag can be used to determine "
        "how the deviation around the group means is plotted.")

    plot_parser.add_argument("--deviation_min", type=float, default=-np.inf,
        help="This flag can be used in conjunction with --group_deviation to "
        "limit the lower end of the plotted deviation.")

    plot_parser.add_argument("--deviation_max", type=float, default=np.inf,
        help="This flag can be used in conjunction with --group_deviation to "
        "limit the upper end of the plotted deviation.")

    plot_parser.add_argument("--verbose", action="store_true",
        help="Enable verbosity.")

    plot_parser.add_argument("--floor", type=float, default=-np.inf,
        help="Only plot curves that have the following characterstic: <floor> "
        "is exceeded at least once within the curve, AND, once <floor> has been "
        "exceeded, the curve never drops below <floor>.")

    plot_parser.add_argument("--ceiling", type=float, default=np.inf,
        help="Only plot curves that have the following characterstic: the "
        "curve drops below <ceil> at least once, AND, once the curve is "
        "below <ceil>, it never exceeds <ceil> again.")

    plot_parser.add_argument("--top", type=int, default=0,
        help="If > 0, only plot the highest <top> curves. Each curve is "
        "reduced along the x-axis before comparisons are made, and the "
        "reduction technique used is defined by the --reduce_x_by flag.")

    plot_parser.add_argument("--bottom", type=int, default=0,
        help="If > 0, only plot the lowest <bottom> curves. Each curve is "
        "reduced along the x-axis before comparisons are made, and the "
        "reduction technique used is defined by the --reduce_x_by flag.")

    plot_parser.add_argument("--reduce_x_by", type=str, default="sum",
        choices=["sum", "min", "max"],
        help="How to reduce the x axis of curves before comparisons are made "
        "when using the --top or --bottom flags.")

    args, runner_args = main_parser.parse_known_args()
    arg_dict = vars(args)

    #
    # If we're plotting, that's all we need to do.
    #
    if args.command == "plot":
        #
        # We parse again here because all args should be known. This is just
        # a safety measure.
        #
        main_parser.parse_args()

        #
        # Our default search pattern list includes "" because we want
        # everything.
        #
        inclusive_search_patterns = [""] if args.inclusive_search_patterns is None else args.inclusive_search_patterns
        exclusive_search_patterns = [""] if args.exclusive_search_patterns is None else args.exclusive_search_patterns

        exclude_patterns   = [] if args.exclude_patterns is None else args.exclude_patterns
        status_constraints = args.status_constraints
        group_names        = [] if args.group_names is None else args.group_names

        msg  = f"ERROR: expected status_constraints to be of type dict but "
        msg += f"received type {type(status_constraints)}."
        assert type(status_constraints) == dict, msg

        if not args.grouping and len(group_names) > 0:
            print("ERROR: group_names can only be used when --grouping is True.")
            sys.exit(1)

        plot_curve_files(
            curve_type                = args.curve_type,
            search_paths              = args.curves,
            title                     = args.title,
            inclusive_search_patterns = inclusive_search_patterns,
            exclusive_search_patterns = exclusive_search_patterns,
            exclude_patterns          = exclude_patterns,
            status_constraints        = status_constraints,
            grouping                  = args.grouping,
            group_names               = group_names,
            group_deviation           = args.group_deviation,
            deviation_min             = args.deviation_min,
            deviation_max             = args.deviation_max,
            verbose                   = args.verbose,
            floor                     = args.floor,
            ceil                      = args.ceiling,
            top                       = args.top,
            bottom                    = args.bottom,
            reduce_x_by               = args.reduce_x_by,
            save_path                 = args.save_path)

        return

    elif args.command == "train":
        if len(runner_args) > 0:
            msg  = f"The following args are unrecognized by the "
            msg += f"primary PPOAF parser and will be sent to the runner: "
            msg += f"{runner_args}"
            rank_print(msg)

        arg_dict["random_seed"] = arg_dict["random_seed"] + rank
        random_seed             = arg_dict["random_seed"]
        runner_file             = arg_dict["runner"]
        force_deterministic     = arg_dict["force_deterministic"]

        #
        # We need to add the parameters for saving out curves. These are
        # the inverse of the disabling flags.
        #
        arg_dict["save_train_scores"] = not arg_dict["disable_save_train_scores"]
        arg_dict["save_avg_ep_len"]   = not arg_dict["disable_save_avg_ep_len"]
        arg_dict["save_running_time"] = not arg_dict["disable_save_running_time"]

        arg_dict["freeze_policies"] = [] if arg_dict["freeze_policies"] is None \
            else arg_dict["freeze_policies"]

        arg_dict = update_pretrained(arg_dict)

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

        runner_class = get_runner_class(runner_file)
        file_preface = os.path.basename(runner_file).split('.')[:-1][0]
        arg_dict["state_path"] = os.path.join(
            args.state_path , file_preface)

        if args.state_tag != "":
            arg_dict["state_path"] = f"{arg_dict['state_path']}-{args.state_tag}"

        clobber      = arg_dict["clobber"]
        load_state   = False

        if clobber:
            if rank == 0:
                #
                # Remove the old saves and re-create the directory.
                #
                if os.path.exists(arg_dict["state_path"]):
                    shutil.rmtree(arg_dict["state_path"])

                os.makedirs(arg_dict["state_path"])

        elif os.path.exists(os.path.join(arg_dict["state_path"], "state_0.pickle")):
            load_state = True

        else:
            if rank == 0:
                os.makedirs(arg_dict["state_path"])

        #
        # We copy the runner to the state directory to assist with
        # testing.
        #
        if rank == 0:
            dest_file = os.path.join(arg_dict["state_path"],
                "runner.py")
            shutil.copyfile(runner_file, dest_file)

        comm.barrier()

    elif args.command == "test":

        arg_dict = update_pretrained(arg_dict)

        random_seed = arg_dict["random_seed"]

        if random_seed >= 0:
            torch.manual_seed(random_seed)
            random.seed(random_seed)
            np.random.seed(random_seed)

        if args.render and args.render_gif:
            msg  = "ERROR: render and render_gif are both enabled, "
            msg += "but they cannot be used simultaneously."
            rank_print(msg)
            comm.Abort()

        if not os.path.exists(os.path.join(args.state_path, "state_0.pickle")):
            rank_print("ERROR: unable to find saved state for testing!")
            comm.Abort()

        test_runner_file = os.path.join(args.state_path,
            "runner.py")

        runner_class = get_runner_class(test_runner_file)
        load_state   = True

    #
    # Create our runner class.
    #
    runner = runner_class(
        **arg_dict,
        test       = args.command == "test",
        load_state = load_state)

    #
    # If we're training, we need to send any unknown args to the runner's
    # arg parser and then save out our args to yaml files.
    #
    if args.command == "train":

        #
        # Allow the runner to add more args to the parser if needed and
        # store them internally.
        #
        _, runner_args = runner.parse_extended_cli_args(runner_args)
        runner_arg_dict = vars(runner_args)

        #
        # Save our runner parameters argparse args so that we can easily test
        # from a state directory alone.
        #
        if rank == 0:
            args_file = os.path.join(arg_dict["state_path"], "args.yaml")
            with open(args_file, "w") as out_f:
                yaml.dump(arg_dict, out_f, default_flow_style=False)

            runner_args_file = os.path.join(arg_dict["state_path"], "runner_args.yaml")
            with open(runner_args_file, "w") as out_f:
                yaml.dump(runner_arg_dict, out_f, default_flow_style=False)

        comm.barrier()

    #
    # If we're testing, we need to load in the previous runner args that were
    # used during training. This will allow us to utilize those same args.
    #
    elif args.command == "test":

        runner_args_file = os.path.join(arg_dict["state_path"], "runner_args.yaml")
        with open(runner_args_file, "r") as in_f:
            saved_runner_args = yaml.safe_load(in_f)

        #
        # Tricky business: by default, we load in the runner args that were
        # used during training, and this is what we use for testing. However,
        # if a user issues runner args during testing, we need to check for them
        # and override the previously saved args.
        #
        if len(runner_args) > 0:

            #
            # This is a nice solution that I found here:
            # https://stackoverflow.com/questions/58594956/find-out-which-arguments-were-passed-explicitly-in-argparse
            #
            class Sentinel():
                pass

            sentinel      = Sentinel()
            sentinel_args = {key : sentinel for key in saved_runner_args}
            for key in sentinel_args:
                sentinel_args[key] = sentinel

            sentinel_ns = argparse.Namespace(**sentinel_args)

            runner.parse_extended_cli_args(runner_args, sentinel_ns)

            sentinel_args = vars(sentinel_ns)

            overwritten = []
            for arg in saved_runner_args:
                if sentinel_args[arg] is not sentinel:
                    saved_runner_args[arg] = sentinel_args[arg]
                    overwritten.append(arg)

            if len(overwritten) > 0:
                msg  = f"WARNING: the following runner args are being overwritten: "
                msg += f"{overwritten}"
                rank_print(msg)

        runner_args = saved_runner_args

        #
        # If we're testing, assign the old parser args to the runner's
        # cli args so that we reuse any extra variables that were
        # added when training.
        #
        runner.cli_args = argparse.Namespace(**runner_args)

    #
    # Run!
    #
    runner.run()

if __name__ == "__main__":
    cli()

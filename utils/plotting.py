"""
Plot curve files saved from ppoaf training.
"""
import pickle
import numpy as np
import glob
import os
import plotly.graph_objects as go
import sys
from pathlib import Path
import difflib
import plotly.express as px

comp_str_map = {
    "<"  : np.less,
    ">"  : np.greater,
    "<=" : np.less_equal,
    ">=" : np.greater_equal,
    "="  : np.equal,
}

def get_str_overlap(s1, s2):
    """
    Nice function for finding the overlap between two strings.
    Taken from
    https://stackoverflow.com/questions/14128763/how-to-find-the-overlap-between-2-sequences-and-return-it

    Parameters:
    -----------
    s1: str
        String 1.
    s2: str
        String 2.

    Returns:
    --------
        The overlap between s1 and s2.
    """
    match = difflib.SequenceMatcher(None, s1, s2)
    pos_a, pos_b, size = match.find_longest_match(0, len(s1), 0, len(s2)) 
    return s1[pos_a : pos_a + size]

def get_status_dict(state_path):
    """
    Load that status dictionary from the state path.

    Parameters:
    -----------
    state_path: str
        The full path to saved state from a training.

    Returns:
    --------
    dict:
        The status dictionary from training.
    """

    file_name  = "state_0.pickle"
    state_file = os.path.join(state_path, file_name)

    with open(state_file, "rb") as in_f:
        status_dict = pickle.load(in_f)

    return status_dict

def status_conditions_are_met(status_conditions, status_dict):
    """
    Determine whether or not status conditions are met within
    a particular status diciontary.

    Parameters:
    -----------
    status_conditions: dict
        A diciontary of status conditions. The should map status keys
        to other status keys or tuples. Example:
        {'status_name_0' : ('comp_func_0', comp_val_0), 'status_preface'
        : {'status_name_1' : ('comp_func_1', comp_val_1)}} s.t. 'comp_func_i' is
        one of <, >, <=, >=, =.
    status_dict: dict
        A status dictionary from a saved training.

    Returns:
    --------
    bool:
        Whether or not all conditions are met.
    """

    for key in status_conditions:

        val = status_conditions[key]

        if type(val) == tuple:
            comp_str, comp_val = val
            comp_func = comp_str_map[comp_str.strip()]

            if not comp_func(status_dict[key.strip()], float(comp_val)):
                return False

        else:
            if key not in status_dict:
                msg  = f"ERROR: key, {key}, is not in the status dictionary"
                print(msg)
                return False

            if not status_conditions_are_met(status_conditions[key], status_dict[key.strip()]):
                return False

    return True

def include_plot_file(
    plot_file,
    search_patterns,
    exclude_patterns,
    status_conditions):
    """
    Given search and exclude patterns, should the plot file be included?

    Parameters:
    -----------
    plot_file: str
        A path to a potential file to plot.
    search_patterns: array-like
        Array of strings representing patterns that should be plotted
        (while excluding all others).
    exclude_patterns: array-like
        Array of strings representing patterns that should not be plotted
        (while including all others).
    status_conditions: dict
        A diciontary of status conditions. The should map status keys
        to other status keys or tuples. Example:
        {'status_name_0' : ('comp_func_0', comp_val_0), 'status_preface'
        : {'status_name_1' : ('comp_func_1', comp_val_1)}} s.t. 'comp_func_i' is
        one of <, >, <=, >=, =.

    Returns:
    --------
    bool:
        Whether or not the plot file should be plotted. This will be True
        iff all strings within search_patterns are contained within plot_file
        and all strings within exclude_patterns are NOT contained within
        plot_file.
    """
    for s_p in search_patterns:
        if s_p in plot_file:
            for e_p in exclude_patterns:
                if e_p in plot_file:
                    return False

            #
            # Get status file
            #
            state_path  = Path(plot_file).parent.parent.parent.absolute()
            status_dict = get_status_dict(state_path)
            return status_conditions_are_met(status_conditions, status_dict)

    return False

def find_curve_files(
    curve_dir_name,
    root,
    search_patterns,
    exclude_patterns,
    status_conditions):
    """
    Recursively find all desired curve files from a given path.

    Parameters:
    -----------
    curve_dir_name: str
        The name of the directory containing curve files.
    root: str
        The root directory to recursively search.
    search_patterns: array-like
        Array of strings representing patterns that should be plotted
        (while excluding all others).
    exclude_patterns: array-like
        Array of strings representing patterns that should not be plotted
        (while including all others).
    status_conditions: dict
        A diciontary of status conditions. The should map status keys
        to other status keys or tuples. Example:
        {'status_name_0' : ('comp_func_0', comp_val_0), 'status_preface'
        : {'status_name_1' : ('comp_func_1', comp_val_1)}} s.t. 'comp_func_i' is
        one of <, >, <=, >=, =.

    Returns:
    --------
    list:
        Full paths to all curve files that meet the given contraints.
    """
    curve_files = []
    for path, dirs, files in os.walk(root):
        if curve_dir_name in dirs:
            curve_dir = os.path.join(path, curve_dir_name)

            np_files = os.path.join(curve_dir, "*.npy")
            for dir_file in glob.glob(np_files):
                if include_plot_file(dir_file, search_patterns, exclude_patterns, status_conditions):
                    curve_files.append(dir_file)

    return curve_files


def plot_curves_with_plotly(
    curve_files,
    add_markers = False):
    """
    Plot a list of curve files with plotly.

    Parameters:
    -----------
    curve_files: array-like
        An array/list of numpy txt files containing curves to plot.
    add_markers: bool
        Should we add markers to our lines?
    """
    curves      = []
    curve_names = []
    for cf in curve_files:
        path_parts = cf.split(os.sep)
        test_name  = path_parts[-4]
        curve_name = path_parts[-1]
        curve_name = " ".join(curve_name.split(".")[0].split("_"))
        name       = f"{test_name} {curve_name}"

        curve_names.append(name)
        with open(cf, "rb") as in_f:
            curves.append(np.loadtxt(in_f))

    fig = go.Figure()

    mode = "lines"
    if add_markers:
        mode = f"{mode}+markers"

    for i in range(len(curve_names)):
        iterations = np.arange(curves[i].size)
        fig.add_trace(
            go.Scatter(
                x      = iterations,
                y      = curves[i],
                mode   = mode,
                name   = curve_names[i]))
    
    fig.show()

def plot_grouped_curves_with_plotly(
    curve_files,
    group_names = [],
    add_markers = False,
    verbose     = False):
    """
    Plot a list of curve file gruops with plotly. The std and mean of
    each group will be plotted.

    Parameters:
    -----------
    curve_files: array-like
        An array/list of lists containing numpy txt files containing curves
        to plot. Each sub-list is considered a group.
    group_names: list
        An optional list of group names. If empty, a name will be auto-generated.
        If not empty, there must be a name for every group.
    add_markers: bool
        Should we add markers to our lines?
    verbose: bool
        Enable verbosity?
    """
    fig    = go.Figure()
    colors = px.colors.qualitative.Plotly

    mean_mode = "lines"
    if add_markers:
        mean_mode = f"{mean_mode}+markers"

    auto_group_name = True
    if len(group_names) > 0:
        msg  = "ERROR: when defining group_names, there must be a name "
        msg += f"for every group. Found {len(group_names)} group names "
        msg += f"and {len(curve_files)} groups."
        assert len(curve_files) == len(group_names), msg

        auto_group_name = False

    for g_idx, group in enumerate(curve_files):
        curves      = []
        group_color = colors[g_idx]

        if auto_group_name:
            group_name = None
        else:
            group_name = group_names[g_idx]

        for cf in group:
            path_parts = cf.split(os.sep)
            test_name  = path_parts[-4]
            curve_name = path_parts[-1]
            curve_name = " ".join(curve_name.split(".")[0].split("_"))
            name       = f"{test_name} {curve_name}"

            if auto_group_name:
                if group_name is None:
                    group_name = name
                else:
                    group_name = get_str_overlap(group_name, name)

            with open(cf, "rb") as in_f:
                curves.append(np.loadtxt(in_f))

        x_size = curves[0].size
        for c in curves:
            msg  = "ERROR: grouped curves must all have the same number "
            msg += "of iterations."
            assert x_size == c.size, msg

        if auto_group_name and group_name == "":
            group_name = f"group_{g_idx}"
            msg  = "WARNING: unable to find overlapping group name. "
            msg += f"Defaulting to generic name '{group_name}'."
            print(msg)

        if verbose:
            print(f"Adding group {group_name} with files {group}")

        curve_stack = np.stack(curves)
        std_min     = curve_stack.min(axis=0)
        std_max     = curve_stack.max(axis=0)
        mean        = curve_stack.mean(axis=0)

        iterations = np.arange(x_size)

        fig.add_trace(
            go.Scatter(
                x          = iterations,
                y          = mean,
                line       = dict(color=group_color),
                mode       = mean_mode,
                name       = group_name))

        std_name = f"{group_name}_std"
        fig.add_trace(
            go.Scatter(
                x          = np.concatenate([iterations, iterations[::-1]]),
                y          = np.concatenate([std_max, std_min[::-1]]),
                fill       = 'toself',
                fillcolor  = group_color,
                mode       = 'none',
                opacity    = 0.2,
                showlegend = False,
                name       = std_name))
    
    fig.show()

def plot_curve_files(
    curve_type,
    search_paths,
    search_patterns,
    exclude_patterns,
    status_conditions,
    add_markers = False,
    grouping    = False,
    group_names = [],
    verbose     = False):
    """
    Plot any number of curve files using plotly.

    Parameters:
    -----------
    curve_type: str
        The name of the curve type to search for. For instance, "scores" will
        result in searching for scores. These curve types will be located in
        <state_path>/curves/. A the time of writing this, curve types are
        "scores", "episode_length", and "runtime".
    search_paths: array-like
        Paths to the policy curve files that you wish to plot. This can be paths
        to the actual curve files, directories containing the curve files,
        or directories containing sub-directories (at any depth) containing
        curve files. These curve files are numpy txt files.
    search_patterns: array-like
        Only plot files that contain these strings within their paths.
    exclude_patterns: array-like
        Only plot files that don't contain these strings within their paths.
    status_conditions: dict
        A diciontary of status conditions. The should map status keys
        to other status keys or tuples. Example:
        {'status_name_0' : ('comp_func_0', comp_val_0), 'status_preface'
        : {'status_name_1' : ('comp_func_1', comp_val_1)}} s.t. 'comp_func_i' is
        one of <, >, <=, >=, =.
    add_markers: bool
        If True, add markers to the line plots.
    grouping: bool
        If grouping is True, curves will be grouped together
        by their search paths. The std and mean of each group will be plotted.
    group_names: list
        An optional list of group names. If empty, a name will be auto-generated.
        If not empty, there must be a name for every group.
        Only applicable when grouping == True.
    verbose: bool
        Enable verbosity?
    """
    curve_files = []
    for sp in search_paths:
        if sp.endswith(".npy"):
            if include_plot_file(sp, search_patterns, exclude_patterns, status_conditions):
                if grouping:
                    curve_files.append([sp])
                else:
                    curve_files.append(sp)
        else:
            path_files = find_curve_files(
                curve_type,
                sp,
                search_patterns,
                exclude_patterns,
                status_conditions)

            if grouping:
                curve_files.append(path_files)
            else:
                curve_files.extend(path_files)

    print(f"Found the following curve files: \n{curve_files}")
    if len(curve_files) == 0:
        sys.exit()

    if grouping:
        plot_grouped_curves_with_plotly(
            curve_files = curve_files,
            group_names = group_names,
            add_markers = add_markers,
            verbose     = verbose)
    else:
        plot_curves_with_plotly(
            curve_files = curve_files,
            add_markers = add_markers)


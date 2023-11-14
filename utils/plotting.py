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

comp_str_map = {
    "<"  : np.less,
    ">"  : np.greater,
    "<=" : np.less_equal,
    ">=" : np.greater_equal,
    "="  : np.equal,
}

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


def plot_curves_with_plotly(curve_files):
    """
    Plot a list of curve files with plotly.

    Parameters:
    -----------
    curve_files: array-like
        An array/list of numpy txt files containing curves to plot.
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

    for i in range(len(curve_names)):
        iterations = np.arange(curves[i].size)
        fig.add_trace(go.Scatter(x=iterations, y=curves[i],
                            mode='lines+markers',
                            name=curve_names[i]))
    
    fig.show()

def plot_curve_files(
    curve_type,
    search_paths,
    search_patterns,
    exclude_patterns,
    status_conditions):
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
    """
    curve_files = []
    for sp in search_paths:
        if sp.endswith(".npy"):
            if include_plot_file(sp, search_patterns, exclude_patterns, status_conditions):
                curve_files.append(sp) 
        else:
            curve_files.extend(find_curve_files(curve_type, sp, search_patterns, exclude_patterns, status_conditions))

    print(f"Found the following curve files: \n{curve_files}")
    if len(curve_files) == 0:
        sys.exit()

    plot_curves_with_plotly(curve_files)

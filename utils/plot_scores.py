"""
Plot save extrinsice score average files for policies.
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

            return status_conditions_are_met(
                status_conditions[key],
                status_dict[key.strip()])

    return True

def include_plot_file(plot_file, search_patterns, exclude_patterns, status_conditions):
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
            state_path  = Path(plot_file).parent.parent.absolute()
            status_dict = get_status_dict(state_path)
            return status_conditions_are_met(status_conditions, status_dict)

    return False

def find_score_files(score_dir_name, root, search_patterns, exclude_patterns, status_conditions):
    score_files = []
    for path, dirs, files in os.walk(root):
        if score_dir_name in dirs:
            score_dir = os.path.join(path, score_dir_name)

            np_files = os.path.join(score_dir, "*.npy")
            for dir_file in glob.glob(np_files):
                if include_plot_file(dir_file, search_patterns, exclude_patterns, status_conditions):
                    score_files.append(dir_file)

    return score_files

def plot_score_files(
    search_paths,
    search_patterns,
    exclude_patterns,
    status_conditions):
    """
    Plot any number of score files using plotly.

    Parameters:
    -----------
    search_paths: array-like
        Paths to the policy score files that you wish to plot. This can be paths
        to the actual score files, directories containing the score files,
        or directories containing sub-directories (at any depth) containing
        score files.
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
    score_files = []
    for sp in search_paths:
        if sp.endswith(".npy"):
            if include_plot_file(sp, search_patterns, exclude_patterns, status_conditions):
                score_files.append(sp) 
        else:
            score_files.extend(find_score_files("scores", sp, search_patterns, exclude_patterns, status_conditions))

    print(f"Found the following score files: \n{score_files}")
    if len(score_files) == 0:
        sys.exit()

    score_arrays = []
    score_names  = []
    for sf in score_files:
        path_parts = sf.split(os.sep)
        test_name  = path_parts[-3]
        score_name = path_parts[-1]
        score_name = " ".join(score_name.split(".")[0].split("_"))
        name       = f"{test_name} {score_name}"

        score_names.append(name)
        with open(sf, "rb") as in_f:
            score_arrays.append(np.loadtxt(in_f))

    fig = go.Figure()

    for i in range(len(score_names)):
        iterations = np.arange(score_arrays[i].size)
        fig.add_trace(go.Scatter(x=iterations, y=score_arrays[i],
                            mode='lines+markers',
                            name=score_names[i]))
    
    fig.show()

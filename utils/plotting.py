"""
Plot curve files saved from ppoaf training.
"""
import functools
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

def filter_curves_by_floor(curve_files, floor):
    """
    Filter out curve files by a given floor.

    Parameters:
    -----------
    curve_files: array-like
        An array/list of paths to numpy txt files containing curves
        to filter.
    floor: float
        Only plot curves that have the following characterstic: <floor>
        is exceeded at least once within the curve, AND, once <floor> has been
        exceeded, the curve never drops below <floor>.

    Returns:
    --------
    list:
        A list of filtered curve_files.
    """
    keeper_curves = []
    for cf in curve_files:
        with open(cf, "rb") as in_f:
            curve = np.loadtxt(in_f)

            #
            # Make sure that this curve exceeds the floor at least once.
            # if not, scrap it.
            #
            where_greater = np.where(curve > floor)[0]
            if where_greater.size == 0:
                continue

            #
            # Make sure that, once the curve has exceeded floor, it no
            # longer dips below.
            #
            g_start = where_greater[0]
            if np.where(curve[g_start : ] < floor)[0].size == 0:
                keeper_curves.append(cf)

    return keeper_curves

def filter_curves_by_ceil(curve_files, ceil):
    """
    Filter out curve files by a given ceil.

    Parameters:
    -----------
    curve_files: array-like
        An array/list of paths to numpy txt files containing curves
        to filter.
    ceil: float
        Only plot curves that have the following characterstic: the
        curve drops below <ceil> at least once, AND, once the curve is
        below <ceil>, it never exceeds <ceil> again.

    Returns:
    --------
    list:
        A list of filtered curve_files.
    """
    keeper_curves = []
    for cf in curve_files:
        with open(cf, "rb") as in_f:
            curve = np.loadtxt(in_f)

            #
            # Make sure that this curve drops below ceil at least once.
            # if not, scrap it.
            #
            where_greater = np.where(curve < ceil)[0]
            if where_greater.size == 0:
                continue

            #
            # Make sure that, once the curve has dropped below ceil, it no
            # longer exceeds it.
            #
            g_start = where_greater[0]
            if np.where(curve[g_start : ] > ceil)[0].size == 0:
                keeper_curves.append(cf)

    return keeper_curves

def get_curves_from_files(curve_files):
    """
    Load all curves from the given files into a list of numpy arrays.

    Parameters:
    -----------
    curve_files: array-like
        An array/list of paths to numpy txt files containing curves
        to filter.

    Returns:
    --------
    list:
        A list of numpy arrays.
    """
    curves = []
    for cf in curve_files:
        with open(cf, "rb") as in_f:
            curve = np.loadtxt(in_f)
            curves.append(curve)

    return curves

def get_sorted_curve_files(curve_files, reduce_x_by = "sum"):
    """
    Get a version of the curve files that is sorted from lowest to highest
    sum.

    Parameters:
    -----------
    curve_files: array-like
        An array/list of paths to numpy txt files containing curves
        to filter.
    reduce_x_by: str
        How to reduce the curves before comparing them.

    Returns:
    --------
    list:
        A list of sorted curve_files.
    """
    curves  = get_curves_from_files(curve_files)
    reduced = [getattr(np, reduce_x_by)(c) for c in curves]

    sorted_idxs = np.argsort(reduced)
    return np.array(curve_files)[sorted_idxs]

def filter_curves_by_top(curve_files, top, reduce_x_by):
    """
    Filter out curve files by only keeping the highest <top> curves.

    Parameters:
    -----------
    curve_files: array-like
        An array/list of paths to numpy txt files containing curves
        to filter.
    top: int
        After all curves are ranked in descending order, only return the
        highest <top> curves.
    reduce_x_by: str
        How to reduce the curves before comparing them.

    Returns:
    --------
    list:
        A list of filtered curve_files.
    """
    sorted_cf = get_sorted_curve_files(curve_files, reduce_x_by)
    sorted_cf = np.flip(sorted_cf)

    return sorted_cf[:top]

def filter_curves_by_bottom(curve_files, bottom, reduce_x_by):
    """
    Filter out curve files by only keeping the highest <bottom> curves.

    Parameters:
    -----------
    curve_files: array-like
        An array/list of paths to numpy txt files containing curves
        to filter.
    bottom: int
        After all curves are ranked in ascending order, only return the
        lowest <bottom> curves.
    reduce_x_by: str
        How to reduce the curves before comparing them.

    Returns:
    --------
    list:
        A list of filtered curve_files.
    """
    sorted_cf = get_sorted_curve_files(curve_files, reduce_x_by)
    return sorted_cf[:bottom]

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

def status_constraints_are_met(status_constraints, status_dict):
    """
    Determine whether or not status conditions are met within
    a particular status diciontary.

    Parameters:
    -----------
    status_constraints: dict
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

    for key in status_constraints:

        val = status_constraints[key]

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

            if not status_constraints_are_met(status_constraints[key], status_dict[key.strip()]):
                return False

    return True

def file_meets_patterns_and_conditions(
    plot_file,
    inclusive_search_patterns,
    exclusive_search_patterns,
    exclude_patterns,
    status_constraints):
    """
    Does a given plot file meet search patterns, exclude patterns, and
    status conditions?

    Parameters:
    -----------
    plot_file: str
        A path to a potential file to plot.
    inclusive_search_patterns: array-like
        Array of strings representing patterns that should be plotted
        (while excluding all others). These are inclusive, meaning ALL
        of them need to appear in the file path.
    exclusive_search_patterns: array-like
        Array of strings representing patterns that should be plotted
        (while excluding all others). These are exclusive, meaning only
        ONE needs to appear in the file path.
    exclude_patterns: array-like
        Array of strings representing patterns that should not be plotted
        (while including all others).
    status_constraints: dict
        A diciontary of status conditions. The should map status keys
        to other status keys or tuples. Example:
        {'status_name_0' : ('comp_func_0', comp_val_0), 'status_preface'
        : {'status_name_1' : ('comp_func_1', comp_val_1)}} s.t. 'comp_func_i' is
        one of <, >, <=, >=, =.

    Returns:
    --------
    bool:
        Whether or not the plot file should be plotted.
    """
    #
    # Ensure that the file path contains ALL search patterns,
    # contains NO exclude patterns, and meets all status conditions.
    #
    for s_p in inclusive_search_patterns:
        if s_p in plot_file:
            for e_p in exclude_patterns:
                if e_p in plot_file:
                    return False

            #
            # Get status file.
            #
            state_path  = Path(plot_file).parent.parent.parent.absolute()
            status_dict = get_status_dict(state_path)

            if not status_constraints_are_met(status_constraints, status_dict):
                return False
        else:
            return False

    if len(exclusive_search_patterns) == 0:
        return True

    for s_p in exclusive_search_patterns:
        if s_p in plot_file:
            for e_p in exclude_patterns:
                if e_p in plot_file:
                    return False

            #
            # Get status file.
            #
            state_path  = Path(plot_file).parent.parent.parent.absolute()
            status_dict = get_status_dict(state_path)

            return status_constraints_are_met(status_constraints, status_dict)

    return False

def find_curve_files(
    curve_dir_name,
    root,
    inclusive_search_patterns,
    exclusive_search_patterns,
    exclude_patterns,
    status_constraints):
    """
    Recursively find all desired curve files from a given path.

    Parameters:
    -----------
    curve_dir_name: str
        The name of the directory containing curve files.
    root: str
        The root directory to recursively search.
    inclusive_search_patterns: array-like
        Array of strings representing patterns that should be plotted
        (while excluding all others). These are inclusive, meaning ALL
        of them need to appear in the file path.
    exclusive_search_patterns: array-like
        Array of strings representing patterns that should be plotted
        (while excluding all others). These are exclusive, meaning only
        ONE needs to appear in the file path.
    exclude_patterns: array-like
        Array of strings representing patterns that should not be plotted
        (while including all others).
    status_constraints: dict
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
                if file_meets_patterns_and_conditions(
                    dir_file,
                    inclusive_search_patterns,
                    exclusive_search_patterns,
                    exclude_patterns,
                    status_constraints):

                    curve_files.append(dir_file)

    return curve_files


def plot_curves_with_plotly(
    curve_files,
    curve_type  = "",
    title       = "",
    add_markers = False,
    save_path   = ""):
    """
    Plot a list of curve files with plotly.

    Parameters:
    -----------
    curve_files: array-like
        An array/list of numpy txt files containing curves to plot.
    curve_type: str
        The type of curve we're plotting. This will become the y axis label.
    title: str
        The plot title.
    add_markers: bool
        Should we add markers to our lines?
    save_path: str
        Optional path to save a figure to instead of rendering in a window.
        The the file should have an extension that is supported by plotly.
    """
    curves      = []
    timesteps   = []
    curve_names = []
    for cf in curve_files:
        path_parts = cf.split(os.sep)
        test_name  = path_parts[-4]
        curve_name = path_parts[-1]
        curve_name = " ".join(curve_name.split(".")[0].split("_"))
        name       = f"{test_name} {curve_name}"

        curve_names.append(name)
        with open(cf, "rb") as in_f:
            data = np.loadtxt(in_f)

            if len(data.shape) > 1:
                timesteps.append(data[:,0])
                curves.append(data[:,1])
            else:
                curves.append(data)

    if len(timesteps) > 0 and len(timesteps) != len(curves):
        msg  = f"\nWARNING: timestep data found for some but not all curves. "
        msg += f"Resorting to iterations for X axis.\n"
        sys.stderr.write(msg)
        timesteps = []

    fig = go.Figure()

    mode = "lines"
    if add_markers:
        mode = f"{mode}+markers"

    for i in range(len(curve_names)):

        if len(timesteps) > 0:
            x_data = timesteps[i]
        else:
            x_data = np.arange(curves[i].size)

        fig.add_trace(
            go.Scatter(
                x      = x_data,
                y      = curves[i],
                mode   = mode,
                name   = curve_names[i]))

    x_title = ""
    if len(timesteps) > 0:
        x_title = "Timesteps"
    else:
        x_title = "Iterations"

    fig.update_layout(
        xaxis_title = x_title,
        yaxis_title = curve_type,
        title       = title,
        title_x     = 0.5,
    )
    
    if save_path != "":
        fig.write_image(save_path)
    else:
        fig.show()

def plot_grouped_curves_with_plotly(
    curve_files,
    group_names   = [],
    curve_type    = "",
    title         = "",
    add_markers   = False,
    deviation     = "std",
    deviation_min = -np.inf,
    deviation_max = np.inf,
    save_path     = "",
    verbose       = False):
    """
    Plot a list of curve file gruops with plotly. The deviation and mean of
    each group will be plotted.

    Parameters:
    -----------
    curve_files: array-like
        An array/list of lists containing numpy txt files containing curves
        to plot. Each sub-list is considered a group.
    group_names: list
        An optional list of group names. If empty, a name will be auto-generated.
        If not empty, there must be a name for every group.
    curve_type: string
        The type of curve we're plotting. This will become the y axis label.
    title: str
        The plot title.
    add_markers: bool
        Should we add markers to our lines?
    deviation: str
        How should the deviation around the mean be plotted?
    deviation_min: float
        The minimum deviation to plot.
    deviation_max: float
        The maximum deviation to plot.
    save_path: str
        Optional path to save a figure to instead of rendering in a window.
        The the file should have an extension that is supported by plotly.
    verbose: bool
        Enable verbosity?
    """
    avail_dev = ["min_max", "std"]
    msg  = f"ERROR: deviation must be one of {avail_dev} but received "
    msg += f"{deviation}."
    assert deviation in avail_dev, msg

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
        if len(group) == 0:
            print(f"Group at index {g_idx} is empty. Skipping...")
            continue

        timesteps   = []
        curves      = []
        curve_names = []
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

            curve_names.append(curve_name)

            if auto_group_name:
                if group_name is None:
                    group_name = name
                else:
                    group_name = get_str_overlap(group_name, name)

            with open(cf, "rb") as in_f:
                data = np.loadtxt(in_f)

                if len(data.shape) > 1:
                    timesteps.append(data[:,0])
                    curves.append(data[:,1])
                else:
                    curves.append(data)

        if auto_group_name and group_name == "":
            group_name = f"group_{g_idx}"
            msg  = "\nWARNING: unable to find overlapping group name. "
            msg += f"Defaulting to generic name '{group_name}'.\n"
            sys.stderr.write(msg)

        if len(timesteps) > 0:
            if len(timesteps) != len(curves):
                msg  = f"\nWARNING: timestep data found for some but not all curves. "
                msg += f"Resorting to iterations for X axis.\n"
                sys.stderr.write(msg)
                timesteps = []
            else:
                msg  = f"ERROR: grouping can only occur if the timesteps between curves "
                msg += f"are identical."
                for i in range(1, len(timesteps)):
                    assert (timesteps[i-1] == timesteps[i]).all(), msg

        if len(timesteps) > 0:
            timesteps = timesteps[0]

        x_size = curves[0].size
        failed_size_check = False
        for c in curves:
            if x_size != c.size:
                msg  = "\nERROR: grouped curves must all have the same number "
                msg += f"of values, but group {group_name} does not."
                msg += f"\ncurves sizes: "

                for i in range(len(curve_names)):
                    msg += f"\n    {curve_names[i]}: {curves[i].size}"

                sys.stderr.write(msg) 
                sys.exit(1)

        curve_stack = np.stack(curves)

        dev_min = None
        dev_max = None
        mean    = curve_stack.mean(axis=0)

        if deviation == "min_max":
            dev_min = np.clip(curve_stack.min(axis=0), deviation_min, np.inf)
            dev_max = np.clip(curve_stack.max(axis=0), -np.inf, deviation_max)
        elif deviation == "std":
            std     = curve_stack.std(axis=0)
            dev_min = np.clip(mean - std, deviation_min, np.inf)
            dev_max = np.clip(mean + std, -np.inf, deviation_max)

        if len(timesteps) > 0:
            x_data = timesteps
        else:
            x_data = np.arange(x_size)

        fig.add_trace(
            go.Scatter(
                x          = x_data,
                y          = mean,
                line       = dict(color=group_color),
                mode       = mean_mode,
                name       = group_name))

        std_name = f"{group_name}_std"
        fig.add_trace(
            go.Scatter(
                x          = np.concatenate([x_data, x_data[::-1]]),
                y          = np.concatenate([dev_max, dev_min[::-1]]),
                fill       = 'toself',
                fillcolor  = group_color,
                mode       = 'none',
                opacity    = 0.2,
                showlegend = False,
                name       = std_name))

    x_title = ""
    if len(timesteps) > 0:
        x_title = "Timesteps"
    else:
        x_title = "Iterations"

    fig.update_layout(
        xaxis_title = x_title,
        yaxis_title = curve_type,
        title       = title,
        title_x     = 0.5,
    )

    if save_path != "":
        fig.write_image(save_path)
    else:
        fig.show()

def filter_grouped_curve_files_by_scores(
    curve_files,
    floor,
    ceil,
    top,
    bottom,
    reduce_x_by):
    """
    Filter grouped curve files by their scores.

    Parameters:
    -----------
    curve_files: array-like
        An array/list of paths to numpy txt files containing curves
        to filter.
    floor: float
        Only plot curves that have the following characterstic: <floor>
        is exceeded at least once within the curve, AND, once <floor> has been
        exceeded, the curve never drops below <floor>.
    ceil: float
        Only plot curves that have the following characterstic: the
        curve drops below <ceil> at least once, AND, once the curve is
        below <ceil>, it never exceeds <ceil> again.
    top: int
        If > 0, only plot the highest <top> curves. Each curve is
        summed along the x axis before comparisons are made.
    bottom: int
        If > 0, only plot the lowest <bottom> curves. Each curve is
        summed along the x axis before comparisons are made.
    reduce_x_by: str
        How to reduce the curves before comparing them.

    Returns:
    --------
    list:
        A list of lists of curve files filtered by their scores.
    """
    filtered_curve_files = []

    for group in curve_files:
        filtered_curve_files.append(filter_curves_by_floor(group, floor))

    curve_files = filtered_curve_files
    filtered_curve_files = []

    for group in curve_files:
        filtered_curve_files.append(filter_curves_by_ceil(group, ceil))

    if top > 0:
        curve_files = filtered_curve_files
        filtered_curve_files = []

        for group in curve_files:
            filtered_curve_files.append(
                filter_curves_by_top(group, top, reduce_x_by))

    if bottom > 0:
        curve_files = filtered_curve_files
        filtered_curve_files = []

        for group in curve_files:
            filtered_curve_files.append(
                filter_curves_by_bottom(group, bottom, reduce_x_by))

    return filtered_curve_files

def filter_curve_files_by_scores(
    curve_files,
    floor,
    ceil,
    top,
    bottom,
    reduce_x_by):
    """
    Filter curve files by their scores.

    Parameters:
    -----------
    curve_files: array-like
        An array/list of paths to numpy txt files containing curves
        to filter.
    floor: float
        Only plot curves that have the following characterstic: <floor>
        is exceeded at least once within the curve, AND, once <floor> has been
        exceeded, the curve never drops below <floor>.
    ceil: float
        Only plot curves that have the following characterstic: the
        curve drops below <ceil> at least once, AND, once the curve is
        below <ceil>, it never exceeds <ceil> again.
    top: int
        If > 0, only plot the highest <top> curves. Each curve is
        summed along the x axis before comparisons are made.
    bottom: int
        If > 0, only plot the lowest <bottom> curves. Each curve is
        summed along the x axis before comparisons are made.
    reduce_x_by: str
        How to reduce the curves before comparing them.

    Returns:
    --------
    list:
        A list of curve files filtered by their scores.
    """
    curve_files = filter_curves_by_floor(curve_files, floor)
    curve_files = filter_curves_by_ceil(curve_files, ceil)

    if top > 0:
        curve_files = filter_curves_by_top(curve_files, top, reduce_x_by)

    if bottom > 0:
        curve_files = filter_curves_by_bottom(curve_files, bottom, reduce_x_by)

    return curve_files

def plot_curve_files(
    curve_type,
    search_paths,
    inclusive_search_patterns,
    exclusive_search_patterns,
    exclude_patterns,
    status_constraints,
    title                     = "",
    add_markers               = False,
    grouping                  = False,
    group_names               = [],
    group_deviation           = "std",
    deviation_min             = -np.inf,
    deviation_max             = np.inf,
    floor                     = -np.inf,
    ceil                      = np.inf,
    top                       = 0,
    bottom                    = 0,
    reduce_x_by               = "sum",
    save_path                 = "",
    verbose                   = False):
    """
    Plot any number of curve files using plotly.

    Parameters:
    -----------
    curve_type: str
        The name of the curve type to search for. For instance, "scores" will
        result in searching for scores. These curve types will be located in
        <state_path>/curves/. A the time of writing this, curve types are
        "scores", "episode_length", "runtime", "bs_min", "bs_max", "bs_avg".
    search_paths: array-like
        Paths to the policy curve files that you wish to plot. This can be paths
        to the actual curve files, directories containing the curve files,
        or directories containing sub-directories (at any depth) containing
        curve files. These curve files are numpy txt files.
    inclusive_search_patterns: array-like
        Only plot files that contain these strings within their paths.
        (while excluding all others). These are inclusive, meaning ALL
        of them need to appear in the file path.
    exclusive_search_patterns: array-like
        Only plot files that contain these strings within their paths.
        (while excluding all others). These are exclusive, meaning only
        ONE needs to appear in the file path.
    exclude_patterns: array-like
        Only plot files that don't contain these strings within their paths.
    status_constraints: dict
        A diciontary of status conditions. The should map status keys
        to other status keys or tuples. Example:
        {'status_name_0' : ('comp_func_0', comp_val_0), 'status_preface'
        : {'status_name_1' : ('comp_func_1', comp_val_1)}} s.t. 'comp_func_i' is
        one of <, >, <=, >=, =.
    title: str
        The plot title to use.
    add_markers: bool
        If True, add markers to the line plots.
    grouping: bool
        If grouping is True, curves will be grouped together
        by their search paths. The deviation and mean of each group will be plotted.
    group_names: list
        An optional list of group names. If empty, a name will be auto-generated.
        If not empty, there must be a name for every group.
        Only applicable when grouping == True.
    group_deviation: str
        How should the deviation around the mean be plotted?
    deviation_min: float
        The minimum deviation to plot.
    deviation_max: float
        The maximum deviation to plot.
    floor: float
        Only plot curves that have the following characterstic: <floor>
        is exceeded at least once within the curve, AND, once <floor> has been
        exceeded, the curve never drops below <floor>.
    ceil: float
        Only plot curves that have the following characterstic: the
        curve drops below <ceil> at least once, AND, once the curve is
        below <ceil>, it never exceeds <ceil> again.
    top: int
        If > 0, only plot the highest <top> curves. Each curve is
        summed along the x axis before comparisons are made.
    bottom: int
        If > 0, only plot the lowest <bottom> curves. Each curve is
        summed along the x axis before comparisons are made.
    reduce_x_by: str
        How to reduce the x axis of curves before comparisons are made
        when using the <top> or <bottom> args.
    save_path: str
        Optional path to save a figure to instead of rendering in a window.
        The the file should have an extension that is supported by plotly.
    verbose: bool
        Enable verbosity?
    """
    curve_files = []
    for sp in search_paths:
        if sp.endswith(".npy"):
            if file_meets_patterns_and_conditions(
                sp,
                inclusive_search_patterns,
                exclusive_search_patterns,
                exclude_patterns,
                status_constraints):

                if grouping:
                    curve_files.append([sp])
                else:
                    curve_files.append(sp)
        else:
            path_files = find_curve_files(
                curve_type,
                sp,
                inclusive_search_patterns,
                exclusive_search_patterns,
                exclude_patterns,
                status_constraints)

            if grouping:
                curve_files.append(path_files)
            else:
                curve_files.extend(path_files)

    if verbose:
        print(f"Found the following curve files: \n{curve_files}")
    else:
        if grouping:
            num_files = sum(len(l) for l in curve_files)
        else:
            num_files = len(curve_files)

        print(f"Found {num_files} curve files")

    if len(curve_files) == 0:
        sys.exit()

    if grouping:

        curve_files = filter_grouped_curve_files_by_scores(
            curve_files = curve_files,
            floor       = floor,
            ceil        = ceil,
            top         = top,
            bottom      = bottom,
            reduce_x_by = reduce_x_by)

        if verbose:
            print(f"Curve files filtered down to: \n{curve_files}")
        else:
            num_files = sum(len(l) for l in curve_files)
            print(f"Curve files filtered down to {num_files}")

        plot_grouped_curves_with_plotly(
            curve_files    = curve_files,
            group_names    = group_names,
            curve_type     = curve_type,
            title          = title,
            add_markers    = add_markers,
            deviation      = group_deviation,
            deviation_min  = deviation_min,
            deviation_max  = deviation_max,
            save_path      = save_path,
            verbose        = verbose)
    else:

        curve_files = filter_curve_files_by_scores(
            curve_files = curve_files,
            floor       = floor,
            ceil        = ceil,
            top         = top,
            bottom      = bottom,
            reduce_x_by = reduce_x_by)

        if verbose:
            print(f"Curve files filtered down to: \n{curve_files}")
        else:
            print(f"Curve files filtered down to {len(curve_files)}")

        plot_curves_with_plotly(
            curve_files = curve_files,
            curve_type  = curve_type,
            title       = title,
            add_markers = add_markers,
            save_path   = save_path)


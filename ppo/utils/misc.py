import numpy as np

def get_action_type(env):
    if np.issubdtype(env.action_space.dtype, np.floating):
        return "continuous"
    elif np.issubdtype(env.action_space.dtype, np.integer):
        return "discrete"
    return "unknown"

def need_action_squeeze(env):
    if np.issubdtype(env.action_space.dtype, np.floating):
        act_dim = env.action_space.shape[0]
    elif np.issubdtype(env.action_space.dtype, np.integer):
        act_dim = env.action_space.n

    action_type = get_action_type(env)

    #
    # Environments are very inconsistent! We need to check what shape
    # they expect actions to be in.
    #
    need_action_squeeze = True
    if action_type == "continuous":
        try:
            padded_shape = (1, act_dim)
            action = np.random.randint(0, 1, padded_shape)
    
            env.reset()
            env.step(action)
            env.reset()
            need_action_squeeze = False
    
        except:
            action_shape = (act_dim,)
            action = np.random.randint(0, 1, action_shape)
    
            env.reset()
            env.step(action)
            env.reset()
            need_action_squeeze = True

    return need_action_squeeze

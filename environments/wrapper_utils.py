from ppo_and_friends.environments.general_wrappers import VectorizedEnv, MultiAgentWrapper
from ppo_and_friends.environments.filter_wrappers import ObservationNormalizer, ObservationClipper
from ppo_and_friends.environments.filter_wrappers import RewardNormalizer, RewardClipper
from ppo_and_friends.environments.filter_wrappers import ObservationAugmentingWrapper
from ppo_and_friends.utils.mpi_utils import rank_print
from collections.abc import Iterable

def wrap_environment(
    env_generator,
    is_multi_agent        = False, #FIXME: remove in future?
    add_agent_ids         = False,
    death_mask            = True,
    envs_per_proc         = 1,
    random_seed           = 2,
    obs_augment           = False,
    normalize_obs         = True,
    normalize_rewards     = True,
    obs_clip              = None,
    reward_clip           = None,
    gamma                 = 0.99,
    test_mode             = False):
    """
    """
    #
    # Begin adding wrappers. Order matters!
    # The first wrapper will always be either a standard vectorization
    # or a multi-agent wrapper. We currently don't support combining them.
    #
    # FIXME: after this transition, all environments will look like multi
    # agent envs (dictionaries). So, we'll need to figure out how to best
    # vectorize them.
    # We should always vectorize our enviornments and return arrays of
    # dictionaries. We can start by only allowing num_envs == 1 for
    # testing/debugging.
    #
    #if is_multi_agent:
    #    env = MultiAgentWrapper(
    #        env_generator  = env_generator,
    #        need_agent_ids = add_agent_ids,
    #        death_mask     = death_mask,
    #        test_mode      = test_mode)
    #else:
    #    env = VectorizedEnv(
    #        env_generator = env_generator,
    #        num_envs      = envs_per_proc,
    #        test_mode     = test_mode)

    env = VectorizedEnv(
        env_generator = env_generator,
        num_envs      = envs_per_proc,
        test_mode     = test_mode)

    #
    # For reproducibility, we need to set the environment's random
    # seeds. Let's allow testing to be random.
    #
    if not test_mode:
        env.set_random_seed(random_seed)

    #
    # The second wrapper should always be the augmenter. This is because
    # our environment should receive pre-normalized data for augmenting.
    #
    if obs_augment:
        if is_multi_agent:
            msg  = "ERROR: observation augmentations are not currently "
            msg += "supported within multi-agent environments."
            rank_print(msg)
            comm.Abort()

        env = ObservationAugmentingWrapper(
            env,
            test_mode = test_mode)

    if normalize_obs:
        env = ObservationNormalizer(
            env          = env,
            test_mode    = test_mode,
            update_stats = not test_mode)

    #FIXME: the status dict in ppo should come from this function.
    status_dict = {}
    if obs_clip != None and type(obs_clip) == tuple:
        env = ObservationClipper(
            env         = env,
            test_mode   = test_mode,
            status_dict = status_dict,
            clip_range  = obs_clip)

    #
    # There are multiple ways to go about normalizing rewards.
    # The approach in arXiv:2006.05990v1 is to normalize before
    # sending targets to the critic and then de-normalize when predicting.
    # We're taking the OpenAI approach of normalizing the rewards straight
    # from the environment and keeping them normalized at all times.
    #
    if normalize_rewards:
        env = RewardNormalizer(
            env          = env,
            test_mode    = test_mode,
            update_stats = not test_mode,
            gamma        = gamma)

    if reward_clip != None and type(reward_clip) == tuple:
        env = RewardClipper(
            env         = env,
            test_mode   = test_mode,
            status_dict = status_dict,
            clip_range  = reward_clip)

    return env, status_dict

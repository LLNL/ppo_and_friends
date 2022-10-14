"""
    A home for generic environment wrappers. The should not be
    specific to any type of environment.
"""
from ppo_and_friends.utils.stats import RunningMeanStd
import numpy as np
import pickle
import os
from ppo_and_friends.utils.mpi_utils import rank_print
from ppo_and_friends.utils.misc import need_action_squeeze
from collections.abc import Iterable
from gym.spaces import Box, Discrete, MultiDiscrete, MultiBinary
from gym.spaces.tuple import Tuple as gym_tuple
from mpi4py import MPI

comm      = MPI.COMM_WORLD
rank      = comm.Get_rank()
num_procs = comm.Get_size()


class IdentityWrapper(object):
    """
        A wrapper that acts exactly like the original environment but also
        has a few extra bells and whistles.
    """

    def __init__(self,
                 env,
                 test_mode = False,
                 **kw_args):
        """
            Initialize the wrapper.

            Arguments:
                env           The environment to wrap.
                status_dict   The dictionary containing training stats.
        """

        self.env               = env
        self.test_mode         = test_mode
        self.observation_space = env.observation_space
        self.action_space      = env.action_space
        self.need_hard_reset   = True
        self.obs_cache         = None
        self.can_augment_obs   = False

        if callable(getattr(self.env, "augment_observation", None)):
            self.can_augment_obs = True

    def get_num_agents(self):
        """
            Get the number of agents in this environment.

            Returns:
                The number of agents in the environment.
        """
        get_num_agents = getattr(self.env, "get_num_agents", None)

        if callable(get_num_agents):
            return get_num_agents()
        return 1

    def get_global_state_space(self):
        """
            Get the global state space. This is specific to multi-agent
            environments. In single agent environments, we just return
            the observation space.

            Returns:
                If available, the global state space is returned. Otherwise,
                the observation space is returned.
        """
        get_global_state_space = getattr(self.env,
            "get_global_state_space", None)

        if callable(get_global_state_space):
            return get_global_state_space()
        return self.observation_space

    def set_random_seed(self, seed):
        """
            Set the random seed for the environment.

            Arguments:
                seed    The seed value.
        """
        self.env.seed(seed)
        self.env.action_space.seed(seed)

    def step(self, action):
        """
            Take a single step in the environment using the given
            action.

            Arguments:
                action    The action to take.

            Returns:
                The resulting observation, reward, done, and info tuple.
        """
        obs, reward, done, info = self.env.step(action)

        self.obs_cache = obs.copy()
        self.need_hard_reset = False

        return obs, reward, done, info

    def reset(self):
        """
            Reset the environment.

            Returns:
                The resulting observation.
        """
        obs = self.env.reset()
        return obs

    def soft_reset(self):
        """
            Perform a "soft reset". This results in only performing the reset
            if the environment hasn't been reset since being created. This can
            allow us to start a new rollout from a previous rollout that ended
            near later in the environments timesteps.

            Returns:
                An observation.
        """
        if self.need_hard_reset or type(self.obs_cache) == type(None):

            #
            # Perform a recursive call on all wrapped environments to check
            # for an ability to soft reset.
            #
            soft_reset = getattr(self.env, "soft_reset", None)

            if callable(soft_reset):
                return soft_reset()

            return self.reset()

        return self.obs_cache

    def render(self, **kw_args):
        """
            Render the environment.
        """
        return self.env.render(**kw_args)

    def save_info(self, path):
        """
            Save any info needed for loading the environment at a later time.

            Arguments:
                path    The path to save to.
        """
        self._check_env_save(path)

    def load_info(self, path):
        """
            Load any info needed for reinstating a saved environment.

            Arguments:
                path    The path to load from.
        """
        self._check_env_load(path)

    def augment_observation(self, obs):
        """
            If our environment has defined an observation augmentation
            method, we can access it here.

            Arguments:
                The observation to augment.

            Returns:
                The batch of augmented observations.
        """
        if self.can_augment_obs:
            return self.env.augment_observation(obs)
        else:
            raise NotImplementedError

    def _check_env_save(self, path):
        """
            Determine if our wrapped environment has a "save_info"
            method. If so, call it.

            Arguments:
                path    The path to save to.
        """
        save_info = getattr(self.env, "save_info", None)

        if callable(save_info):
            save_info(path)

    def _check_env_load(self, path):
        """
            Determine if our wrapped environment has a "load_info"
            method. If so, call it.

            Arguments:
                path    The path to load from.
        """
        load_info = getattr(self.env, "load_info", None)

        if callable(load_info):
            load_info(path)

    def supports_batched_environments(self):
        """
            Determine whether or not our wrapped environment supports
            batched environments.

            Return:
                True or False.
        """
        batch_support = getattr(self.env, "supports_batched_environments", None)

        if callable(batch_support):
            return batch_support()
        else:
            return False

    def get_batch_size(self):
        """
            If any wrapped classes define this method, try to get the batch size
            from them. Otherwise, assume we have a single environment.

            Returns:
                Our batch size.
        """
        batch_size_getter = getattr(self.env, "get_batch_size", None)

        if callable(batch_size_getter):
            return batch_size_getter()
        else:
            return 1

class VectorizedEnv(IdentityWrapper, Iterable):
    """
        Wrapper for "vectorizing" environments. As far as I can tell, this
        definition of vectorize is pretty specific to RL and refers to a way
        of reducing our environment episodes to "fixed-length trajecorty
        segments". This allows us to set our maximum timesteps per episode
        fairly low while still observing an episode from start to finish.
        This is accomplished by these "segments" that can (and often do) start
        in the middle of an episode.
        Vectorized environments are also often referred to as environment
        wrappers that contain multiple instances of environments, which then
        return batches of observations, rewards, etc. We don't currently
        support this second idea.
    """

    def __init__(self,
                 env_generator,
                 num_envs = 1,
                 **kw_args):
        """
            Initialize our vectorized environment.

            Arguments:
                env_generator    A function that creates instances of our
                                 environment.
                num_envs         The number of environments to maintain.
        """
        super(VectorizedEnv, self).__init__(
            env_generator(),
            **kw_args)

        #
        # Environments are very inconsistent! Some of them require their
        # actions to be squeezed before they can be sent to the env.
        #
        self.action_squeeze = need_action_squeeze(self.env)

        self.num_envs = num_envs
        self.envs     = np.array([None] * self.num_envs, dtype=object)
        self.iter_idx = 0

        if self.num_envs == 1:
            self.envs[0] = self.env
        else:
            for i in range(self.num_envs):
                self.envs[i] = env_generator()

    def set_random_seed(self, seed):
        """
            Set the random seed for the environment.

            Arguments:
                seed    The seed value.
        """
        for env_idx in range(self.num_envs):
            self.envs[env_idx].seed(seed)
            self.envs[env_idx].action_space.seed(seed)

    def step(self, action):
        """
            Take a step in our environment with the given action.
            Since we're vectorized, we reset the environment when
            we've reached a "done" state.

            Arguments:
                action    The action to take.
            Returns:
                The resulting observation, reward, done, and info
                tuple.
        """
        #
        # If we're testing, we don't want to return a batch.
        #
        if self.test_mode:
            return self.single_step(action)

        return self.batch_step(action)


    def single_step(self, action):
        """
            Take a step in our environment with the given action.
            Since we're vectorized, we reset the environment when
            we've reached a "done" state.

            Arguments:
                action    The action to take.
            Returns:
                The resulting observation, reward, done, and info
                tuple.
        """
        if self.action_squeeze:
            env_action = action.squeeze()
        else:
            env_action = action

        obs, reward, done, info = self.env.step(env_action)

        #
        # HACK: some environments are buggy and don't follow their
        # own rules!
        #
        obs = obs.reshape(self.observation_space.shape)

        if type(reward) == np.ndarray:
            reward = reward[0]

        if done:
            info["terminal observation"] = obs.copy()
            obs = self.env.reset()
            obs = obs.reshape(self.observation_space.shape)

        return obs, reward, done, info

    def batch_step(self, actions):
        """
            Take a step in our environment with the given actions.
            Since we're vectorized, we reset the environment when
            we've reached a "done" state, and we return a batch
            of step results. Since this is a batch step, we'll return
            a batch of results of size self.num_envs.

            Arguments:
                actions    The actions to take.

            Returns:
                The resulting observation, reward, done, and info
                tuple.
        """
        obs_shape     = (self.num_envs,) + self.observation_space.shape
        batch_obs     = np.zeros(obs_shape)
        batch_rewards = np.zeros((self.num_envs, 1))
        batch_dones   = np.zeros((self.num_envs, 1)).astype(bool)
        batch_infos   = np.array([None] * self.num_envs,
            dtype=object)

        for env_idx in range(self.num_envs):
            act = actions[env_idx]

            if self.action_squeeze:
                act = act.squeeze()

            obs, reward, done, info = self.envs[env_idx].step(act)

            #
            # HACK: some environments are buggy and don't follow their
            # own rules!
            #
            obs = obs.reshape(self.observation_space.shape)

            if type(reward) == np.ndarray:
                reward = reward[0]

            if done:
                info["terminal observation"] = obs.copy()
                obs = self.envs[env_idx].reset()
                obs = obs.reshape(self.observation_space.shape)

            batch_obs[env_idx]     = obs
            batch_rewards[env_idx] = reward
            batch_dones[env_idx]   = done
            batch_infos[env_idx]   = info

        self.obs_cache = batch_obs.copy()

        return batch_obs, batch_rewards, batch_dones, batch_infos

    def reset(self):
        """
            Reset the environment.

            Returns:
                The resulting observation.
        """
        if self.test_mode:
            return self.single_reset()
        return self.batch_reset()

    def single_reset(self):
        """
            Reset the environment.

            Returns:
                The resulting observation.
        """
        obs = self.env.reset()
        return obs

    def batch_reset(self):
        """
            Reset the batch of environments.

            Returns:
                The resulting observation.
        """
        obs_shape = (self.num_envs,) + self.observation_space.shape
        batch_obs = np.zeros(obs_shape)

        for env_idx in range(self.num_envs):
            obs = self.envs[env_idx].reset()
            obs = obs.reshape(self.observation_space.shape)
            batch_obs[env_idx] = obs

        return batch_obs

    def __len__(self):
        """
            Represent our length as our batch size.

            Returns:
                The number of environments in our batch.
        """
        return self.num_envs

    def __next__(self):
        """
            Allow iteration through our array of environments.

            Returns:
                The next environment in our array.
        """
        if self.iter_idx < self.num_envs:
            env = self.envs[self.iter_idx]
            self.iter_idx += 1
            return env

        raise StopIteration

    def __iter__(self):
        """
            Allow iteration through our array of environments.

            Returns:
                Ourself as an iterable.
        """
        return self

    def __getitem__(self, idx):
        """
            Allow accessing environments by index.

            Arguments:
                idx    The index to the desired environment.

            Returns:
                The environment from self.envs located at index idx.
        """
        return self.envs[idx]

    def supports_batched_environments(self):
        """
            Determine whether or not our wrapped environment supports
            batched environments.

            Return:
                True or False.
        """
        return True

    def get_batch_size(self):
        """
            Get the batch size.

            Returns:
                Return our batch size.
        """
        return len(self)


# FIXME: need to handle dictionaries mapping agents to
# spaces.
class MultiAgentWrapper(IdentityWrapper):
    """
        A wrapper for multi-agent environments. This design follows the
        suggestions outlined in arXiv:2103.01955v2.
    """

    def __init__(self,
                 env_generator,
                 need_agent_ids   = True,
                 use_global_state = True,
                 normalize_ids    = True,
                 death_mask       = True,
                 **kw_args):
        """
            Initialize the wrapper.

            Arguments:
                env               The environment to wrap.
                need_agent_ids    Do we need to explicitly add the agent
                                  ids to their observations? Assume yes.
                use_global_state  Send a global state to the critic.
                normalize_ids     If we're explicitly adding ids, should
                                  we first normalize them?
                death_mask        Should we perform death masking?
        """
        super(MultiAgentWrapper, self).__init__(
            env_generator(),
            **kw_args)

        self.need_agent_ids = need_agent_ids
        self.num_agents     = len(self.env.observation_space)
        self.agent_ids      = np.arange(self.num_agents).reshape((-1, 1))

        if normalize_ids:
            self.agent_ids = self.agent_ids / self.num_agents

        self.use_global_state   = use_global_state
        self.global_state_space = None
        self.death_mask         = death_mask

        self.observation_space, _ = self._get_refined_space(
            multi_agent_space = self.env.observation_space,
            add_ids           = need_agent_ids)

        self.action_space, self.num_actions = self._get_refined_space(
            multi_agent_space = self.env.action_space,
            add_ids           = False)

        self._construct_global_state_space()

    def get_num_agents(self):
        """
            Get the number of agents in this environment.

            Returns:
                The number of agents in the environment.
        """
        return self.num_agents

    def get_global_state_space(self):
        """
            Get the global state space.

            Returns:
                self.global_state_space
        """
        return self.global_state_space

    def _get_refined_space(self,
                           multi_agent_space,
                           add_ids = False):
        """
            Given a multi-agent space (action, observation, etc.) consisting
            of tuples of spaces, construct a single space to represent the
            entire environment.

            Some things to note:
              1. In environmens with mixed sized spaces, it's assumed that we
                 will be zero padding the lesser sized spaces. Therefore, the
                 returned space will match the maximum sized space.
              2. Mixing space types is not currently supported.
              3. Discrete and Box are the only two currently supported spaces.

            Arguments:
                multi_agent_space    The reference space. This should consist
                                     of tuples of spaces.
                add_ids              Will the agent ids be appended to this
                                     space? If so, expand the size to account
                                     for this.

            Returns:
                A tuple containing a single space representing all agents as
                the first element and the size of the space as the second
                element.
        """
        #
        # First, ensure that each agent has the same type of space.
        #
        space_type      = None
        prev_space_type = None
        dtype           = None
        prev_dtype      = None
        for space in multi_agent_space:
            if space_type == None:
                space_type      = type(space)
                prev_space_type = type(space)

                dtype         = space.dtype
                prev_dtype    = space.dtype

            if prev_space_type != space_type:
                msg  = "ERROR: mixed space types in multi-agent "
                msg += "environments is not currently supported. "
                msg += "Found types "
                msg += "{} and {}.".format(space_type, prev_space_type)
                rank_print(msg)
                comm.Abort()
            elif prev_dtype != dtype:
                msg  = "ERROR: mixed space dtypes in multi-agent "
                msg += "environments is not currently supported. "
                msg += "Found types "
                msg += "{} and {}.".format(prev_dtype, dtype)
                rank_print(msg)
                comm.Abort()
            else:
                prev_space_type = space_type
                space_type      = type(space)

                prev_dtype    = dtype
                dtype         = space.dtype

        #
        # Next, we need handle cases where agents have different spaces
        # We want to homogenize things, so we pad all of the spaces
        # to be identical.
        #
        # TODO: we're assuming all dimensions are flattened. In the future,
        # we may want to support multi-dimensional spaces.
        #
        low   = np.empty(0, dtype = np.float32)
        high  = np.empty(0, dtype = np.float32)
        count = 0

        for space in multi_agent_space:
            if space_type == Box:
                diff  = space.shape[0] - low.size
                count = max(count, space.shape[0])

                low   = np.pad(low, (0, diff))
                high  = np.pad(high, (0, diff))

                high  = np.maximum(high, space.high)
                low   = np.minimum(low, space.low)

            elif space_type == Discrete:
                count = max(count, space.n)

            else:
                not_supported_msg  = "ERROR: {} is not currently supported "
                not_supported_msg += "as an observation space in multi-agent "
                not_supported_msg += "environments."
                rank_print(not_supported_msg.format(space))
                comm.Abort()

        if space_type == Box:
            if add_ids:
                low   = np.concatenate((low, (0,)))
                high  = np.concatenate((high, (np.inf,)))
                size  = count + 1
                shape = (count + 1,)
            else:
                size  = count

            shape = (size,)

            new_space = Box(
                low   = low,
                high  = high,
                shape = (size,),
                dtype = dtype)

        elif space_type == Discrete:
            size = 1
            if add_ids:
                new_space = Discrete(count + 1)
            else:
                new_space = Discrete(count)

        return (new_space, size)

    def _construct_global_state_space(self):
        """
            Construct the global state space. See arXiv:2103.01955v2.
            By deafult, we concatenate all local observations to represent
            the global state.

            NOTE: this method should only be called AFTER
            _update_observation_space is called.
        """
        #
        # If we're not using the global state space approach, each agent's
        # critic update receives its own observations only.
        #
        if not self.use_global_state:
            self.global_state_space = self.observation_space
            return

        elif hasattr(self.env, "global_state_space"):
            self.global_state_space = self.env.global_state_space
            return

        obs_type = type(self.observation_space)

        if obs_type == Box:
            low   = np.tile(self.observation_space.low, self.num_agents)
            high  = np.tile(self.observation_space.high, self.num_agents)
            shape = (self.observation_space.shape[0] * self.num_agents,)

            global_state_space = Box(
                low   = low,
                high  = high,
                shape = shape,
                dtype = self.observation_space.dtype)

        elif obs_type == Discrete:
            global_state_space = Discrete(
                self.observation_space.n * self.num_agents)
        else:
            not_supported_msg  = "ERROR: {} is not currently supported "
            not_supported_msg += "as an observation space in multi-agent "
            not_supported_msg += "environments."
            rank_print(not_supported_msg.format(self.observation_space))
            comm.Abort()

        self.global_state_space = global_state_space

    def get_feature_pruned_global_state(self, obs):
        """
            From arXiv:2103.01955v2, cacluate the "Feature-Pruned
            Agent-Specific Global State". By default, we assume that
            the environment doesn't offer extra information, which leads
            us to actually caculating the "Concatenation of Local Observations".
            We can create sub-classes of this implementation to handle
            environments that do offer global state info.

            Arguments:
                obs    The refined agent observations.

            Returns:
                The global state to be fed to the critic.
        """
        #
        # If we're not using the global state space approach, each agent's
        # critic update receives its own observations only.
        #
        if not self.use_global_state:
            return obs

        #
        # If our wrapped environment has its own implementation, rely
        # on that. Otherwise, we'll just concatenate all of the agents
        # together.
        #
        pruned_from_env = getattr(self.env,
            "get_feature_pruned_global_state", None)

        if callable(pruned_from_env):
            return pruned_from_env(obs)

        #
        # Our observation is currently in the shape (num_agents, obs), so
        # it's really a batch of observtaions. The global state is really
        # just this batch concatenated as a single observation. Each agent
        # needs a copy of it, so we just tile them. Instead of actually making
        # copies, we can use broadcast_to to create references to the same
        # memory location.
        #
        global_obs = obs.flatten()
        obs_size   = global_obs.size
        global_obs = np.broadcast_to(global_obs, (self.num_agents, obs_size))
        return global_obs

    def _refine_obs(self, obs):
        """
            Refine our multi-agent observations.

            Arguments:
                The multi-agent observations.

            Returns:
                The refined multi-agent observations.
        """
        obs = np.stack(obs, axis=0)

        if self.need_agent_ids:
            obs = np.concatenate((self.agent_ids, obs), axis=1)

        #
        # It's possible for agents to have differing observation space sizes.
        # In those cases, we zero pad for congruity.
        #
        size_diff = self.observation_space.shape[0] - obs.shape[-1]
        if size_diff > 0:
            obs = np.pad(obs, ((0, 0), (0, size_diff)))

        return obs

    def _refine_dones(self, agents_done, obs):
        """
            Refine our done array. Also, perform death masking when
            appropriate. See arXiv:2103.01955v2 for information on
            death masking.

            Arguments:
                agents_done    The done array.
                obs            The agent observations.

            Returns:
                The refined done array as well as a single boolean
                representing whether or not the entire environment is
                done.
        """
        agents_done = np.array(agents_done)

        #
        # We assume that our environment is done only when all agents
        # are done. If death masking is enabled and some but not all
        # agents have died, we need to apply the mask.
        #
        all_done = False
        if agents_done.all():
            all_done = True
        elif self.death_mask:
            obs[agents_done, 1:] = 0.0
            agents_done = np.zeros(self.num_agents).astype(bool)

        agents_done = agents_done.reshape((-1, 1))
        return agents_done, all_done

    def step(self, actions):
        """
            Take a step in our multi-agent environment.

            Arguments:
                actions    The actions for each agent to take.

            Returns:
                observations, rewards, dones, and info.
        """
        #
        # It's possible for agents to have differing action space sizes.
        # In those cases, we zero pad for congruity.
        #
        size_diff = self.num_actions - int(actions.size / self.num_agents)
        if size_diff > 0:
            actions = np.pad(actions, ((0, 0), (0, size_diff)))

        #
        # Our first/test environment expects the actions to be contained in
        # a tuple. We may need to support more formats in the future.
        #
        tup_actions = tuple(actions[i] for i in range(self.num_agents))

        obs, rewards, agents_done, info = self.env.step(actions)

        #
        # Info can come back int two forms:
        #   1. It's a dictionary containing global information.
        #   2. It's an iterable containing num_agent dictionaries,
        #      each of which contains info for its assocated agent.
        #
        info_is_global = False
        if type(info) == dict:
            info_is_global = True
        else:
            info = np.array(info).astype(object)

        obs     = self._refine_obs(obs)
        rewards = np.stack(np.array(rewards), axis=0).reshape((-1, 1))
        dones, all_done = self._refine_dones(agents_done, obs)

        if all_done:
            terminal_obs = obs.copy()
            obs, global_obs = self.reset()

            if info_is_global:
                info["global state"] = global_obs
            else:
                for i in range(info.size):
                    info[i]["global state"] = global_obs

        else:
            global_state = self.get_feature_pruned_global_state(obs)
            if info_is_global:
                info["global state"] = global_state
            else:
                for i in range(info.size):
                    info[i]["global state"] = global_state

        #
        # If our info is global, we need to convert it to local.
        # Create an array of references so that we don't use up memory.
        #
        if info_is_global:
            info = np.array([info] * self.num_agents, dtype=object)

        #
        # Lastly, each agent needs its own terminal observation.
        #
        if all_done:
            for i in range(self.num_agents):
                info[i]["terminal observation"] = terminal_obs[i].copy()

        elif not self.death_mask:
            where_done = np.where(dones)[0]

            for d_idx in where_done:
                info[d_idx]["terminal observation"] = \
                   obs[d_idx].copy()

        self.obs_cache = obs.copy()
        self.need_hard_reset = False

        return obs, rewards, dones, info

    def reset(self):
        """
            Reset the environment. If we're in test mode, we don't augment the
            resulting observations. Otherwise, augment them before returning.

            Returns:
                The local and global observations.
        """
        obs = self.env.reset()
        obs = self._refine_obs(obs)
        global_state = self.get_feature_pruned_global_state(obs)

        return obs, global_state

    def soft_reset(self):
        """
            Perform a "soft reset". This results in only performing the reset
            if the environment hasn't been reset since being created. This can
            allow us to start a new rollout from a previous rollout that ended
            near later in the environments timesteps.

            Returns:
                The local and global observations.
        """
        if self.need_hard_reset or type(self.obs_cache) == type(None):
            return self.reset()

        global_state = self.get_feature_pruned_global_state(self.obs_cache)
        return self.obs_cache, global_state

    def get_batch_size(self):
        """
            Get the batch size.

            Returns:
                Return our batch size.
        """
        return self.num_agents

    def supports_batched_environments(self):
        """
            Determine whether or not our wrapped environment supports
            batched environments.

            Return:
                True.
        """
        return True


class ObservationNormalizer(IdentityWrapper):
    """
        A wrapper for normalizing observations. This normalization
        method uses running statistics.
        NOTE: this uses implentations very similar to some of Stable
        Baslines' VecNormalize.
    """

    def __init__(self,
                 env,
                 update_stats = True,
                 epsilon      = 1e-8,
                 **kw_args):
        """
            Initialize the wrapper.

            Arguments:
                env            The environment to wrap.
                update_stats   Should we update the statistics? We typically
                               have this enabled until testing.
                epsilon        A very small number to help avoid dividing by 0.
        """

        super(ObservationNormalizer, self).__init__(
            env,
            **kw_args)

        self.running_stats = RunningMeanStd(
            shape = self.env.observation_space.shape)

        self.update_stats  = update_stats
        self.epsilon       = epsilon

    def step(self, action):
        """
            Take a single step in the environment using the given
            action, update the running stats, and normalize the
            resulting observation.

            Arguments:
                action    The action to take.

            Returns:
                The resulting observation, reward, done, and info tuple.
        """
        obs, reward, done, info = self.env.step(action)

        if self.update_stats:
            self.running_stats.update(obs)

        if type(obs) == np.ndarray:
            obs = self.normalize(obs.copy())
        else:
            obs = self.normalize(obs)

        return obs, reward, done, info

    def reset(self):
        """
            Reset the environment, and update the running stats.

            Returns:
                The resulting observation.
        """
        obs = self.env.reset()

        #
        # In our multi-agent case, the global state is returned along
        # with the local observation.
        #
        is_multi_agent = False
        if type(obs) == tuple  or type(obs) == list:
            is_multi_agent = True
            obs, global_state = obs

        if self.update_stats:
            self.running_stats.update(obs)

        if is_multi_agent:
            return (obs, global_state)
        return obs

    def normalize(self, obs):
        """
            Normalize an observation using our running stats.

            Arguments:
                obs    The observation to normalize.

            Returns:
                The normalized observation.
        """
        obs = (obs - self.running_stats.mean) / \
            np.sqrt(self.running_stats.variance + self.epsilon)
        return obs

    def save_info(self, path):
        """
            Save out our running stats, and check if our wrapped
            environment needs to perform any more info saves.

            Arguments:
                path    The path to save to.
        """
        if self.test_mode:
            return

        file_name = "RunningObsStats_{}.pkl".format(rank)
        out_file  = os.path.join(path, file_name)

        with open(out_file, "wb") as fh:
            pickle.dump(self.running_stats, fh)

        self._check_env_save(path)

    def load_info(self, path):
        """
            Load our running stats and check to see if our wrapped
            environment needs to load anything.

            Arguments:
                path    The path to load from.
        """
        if self.test_mode:
            file_name = "RunningObsStats_0.pkl"
        else:
            file_name = "RunningObsStats_{}.pkl".format(rank)

        in_file = os.path.join(path, file_name)

        #
        # There are cases where we initially train using X ranks, and we
        # later want to continue training using (X+k) ranks. In these cases,
        # let's copy rank 0's info to all ranks > X.
        #
        if not os.path.exists(in_file):
            file_name = "RunningObsStats_0.pkl"
            in_file   = os.path.join(path, file_name)

        with open(in_file, "rb") as fh:
            self.running_stats = pickle.load(fh)

        self._check_env_load(path)


class RewardNormalizer(IdentityWrapper):
    """
        This wrapper uses running statistics to normalize rewards.
        NOTE: some of this logic comes from Stable Baseline's
        VecNormalize.
    """

    def __init__(self,
                 env,
                 update_stats = True,
                 epsilon      = 1e-8,
                 gamma        = 0.99,
                 **kw_args):
        """
            Initialize the wrapper.

            Arguments:
                env            The environment to wrap.
                update_stats   Whether or not to update our running stats. This
                               is typically set to true until testing.
                epsilon        A very small value to help avoid 0 divisions.
                gamma          A discount factor for our running stats.
        """
        super(RewardNormalizer, self).__init__(
            env,
            **kw_args)

        self.running_stats = RunningMeanStd(shape=())
        self.update_stats  = update_stats
        self.epsilon       = epsilon
        self.gamma         = gamma

        #
        # We might be wrapping an environment that supports batches.
        # if so, we need to be able to correlate a running reward with
        # each environment instance.
        #
        if self.test_mode and self.get_num_agents() == 1:
            self.batch_size = 1
        else:
            self.batch_size = self.get_batch_size()

        self.running_reward = np.zeros(self.batch_size)

    def step(self, action):
        """
            Take a single step in the environment using the given
            action, update the running stats, and normalize the
            resulting reward.

            Arguments:
                action    The action to take.

            Returns:
                The resulting observation, reward, done, and info tuple.
        """

        obs, reward, done, info = self.env.step(action)

        for env_idx in range(self.batch_size):
            if self.update_stats:
                self.running_reward[env_idx] = \
                    self.running_reward[env_idx] * self.gamma + reward[env_idx]

                self.running_stats.update(self.running_reward)

        if self.batch_size > 1:
            where_done = np.where(done)[0]
            self.running_reward[where_done] = 0.0
        elif done:
            self.running_reward[0] = 0.0

        if type(reward) == np.ndarray:
            batch_size = reward.shape[0]

            for r_idx in range(batch_size):
                if "natural reward" not in info[r_idx]:
                    info[r_idx]["natural reward"] = reward[r_idx]

            reward = self.normalize(reward.copy())

        else:
            if "natural reward" not in info:
                info["natural reward"] = reward

            reward = self.normalize(reward)

        return obs, reward, done, info

    def normalize(self, reward):
        """
            Normalize our reward using Stable Baseline's approach.

            Arguments:
                reward    The reward to normalize.

            Returns:
                The normalized reward.
        """
        reward /= np.sqrt(self.running_stats.variance + self.epsilon)
        return reward

    def save_info(self, path):
        """
            Save out our running stats, and check if our wrapped
            environment needs to perform any more info saves.

            Arguments:
                path    The path to save to.
        """
        if self.test_mode:
            return

        file_name = "RunningRewardsStats_{}.pkl".format(rank)
        out_file  = os.path.join(path, file_name)

        with open(out_file, "wb") as fh:
            pickle.dump(self.running_stats, fh)

        self._check_env_save(path)

    def load_info(self, path):
        """
            Load our running stats and check to see if our wrapped
            environment needs to load anything.

            Arguments:
                path    The path to load from.
        """
        if self.test_mode:
            file_name = "RunningRewardsStats_0.pkl"
        else:
            file_name = "RunningRewardsStats_{}.pkl".format(rank)

        in_file = os.path.join(path, file_name)

        #
        # There are cases where we initially train using X ranks, and we
        # later want to continue training using (X+k) ranks. In these cases,
        # let's copy rank 0's info to all ranks > X.
        #
        if not os.path.exists(in_file):
            file_name = "RunningRewardsStats_0.pkl"
            in_file   = os.path.join(path, file_name)

        with open(in_file, "rb") as fh:
            self.running_stats = pickle.load(fh)

        self._check_env_load(path)


class GenericClipper(IdentityWrapper):
    """
        A wrapper for clipping rewards.
    """

    def __init__(self,
                 env,
                 status_dict = {},
                 clip_range  = (-10., 10.),
                 **kw_args):
        """
            Initialize the wrapper.

            Arguments:
                env         The environment to wrap.
                status_dict The training status dictionary. This is used when
                            our clip range contains callables.
                clip_range  The range to clip our rewards into. This can be
                            either a real number or a class that inherits from
                            IterationMapper.
        """

        super(GenericClipper, self).__init__(
            env,
            **kw_args)

        min_callable = None
        max_callable = None
        self.status_dict = status_dict

        if callable(clip_range[0]):
            min_callable = clip_range[0]
        else:
            min_callable = lambda *args, **kwargs : clip_range[0]

        if callable(clip_range[1]):
            max_callable = clip_range[1]
        else:
            max_callable = lambda *args, **kwargs : clip_range[1]

        self.clip_range = (min_callable, max_callable)

    def get_clip_range(self):
        """
            Get the current clip range.

            Returns:
                A tuple containing the clip range as (min, max).
        """
        min_value = self.clip_range[0](
            iteration = self.status_dict["iteration"],
            timestep  = self.status_dict["timesteps"])
        max_value = self.clip_range[1](
            iteration = self.status_dict["iteration"],
            timestep  = self.status_dict["timesteps"])

        return (min_value, max_value)

    def _clip(self, val):
        """
            Perform the clip.

            Arguments:
                val    The value to be clipped.
        """
        raise NotImplementedError


class ObservationClipper(GenericClipper):
    """
        An environment wrapper that clips observations.
    """

    def __init__(self,
                 env,
                 status_dict = {},
                 clip_range  = (-10., 10.),
                 **kw_args):
        """
            Initialize the wrapper.

            Arguments:
                env         The environment to wrap.
                status_dict The training status dictionary. This is used when
                            our clip range contains callables.
                clip_range  The range to clip our observations into.
        """
        super(ObservationClipper, self).__init__(
            env,
            status_dict = status_dict,
            clip_range  = clip_range,
            **kw_args)

    def step(self, action):
        """
            Take a single step in the environment using the given
            action, and clip the resulting observation.

            Arguments:
                action    The action to take.

            Returns:
                The resulting observation, reward, done, and info tuple.
        """
        obs, reward, done, info = self.env.step(action)

        obs = self._clip(obs)

        return obs, reward, done, info

    def reset(self):
        """
            Reset the environment, and clip the observation.

            Returns:
                The resulting observation.
        """
        obs = self.env.reset()

        #
        # In our multi-agent case, the global state is returned along
        # with the local observation.
        #
        is_multi_agent = False
        if type(obs) == type(tuple()):
            is_multi_agent = True
            obs, global_state = obs

        obs = self._clip(obs)

        if is_multi_agent:
            return (obs, global_state)
        return obs

    def _clip(self, obs):
        """
            Perform the observation clip.

            Arguments:
                obs    The observation to clip.

            Returns:
                The clipped observation.
        """
        min_value, max_value = self.get_clip_range()
        return np.clip(obs, min_value, max_value)


class RewardClipper(GenericClipper):
    """
        A wrapper for clipping rewards.
    """

    def __init__(self,
                 env,
                 status_dict = {},
                 clip_range  = (-10., 10.),
                 **kw_args):
        """
            Initialize the wrapper.

            Arguments:
                env         The environment to wrap.
                status_dict The training status dictionary. This is used when
                            our clip range contains callables.
                clip_range  The range to clip our rewards into.
        """

        super(RewardClipper, self).__init__(
            env,
            status_dict = status_dict,
            clip_range  = clip_range,
            **kw_args)

    def step(self, action):
        """
            Take a single step in the environment using the given
            action, and clip the resulting reward.

            Arguments:
                action    The action to take.

            Returns:
                The resulting observation, reward, done, and info tuple.
        """
        obs, reward, done, info = self.env.step(action)

        if type(reward) == np.ndarray:
            batch_size = reward.shape[0]

            for r_idx in range(batch_size):
                if "natural reward" not in info[r_idx]:
                    info[r_idx]["natural reward"] = reward[r_idx]
        else:
            if "natural reward" not in info:
                info["natural reward"] = reward

        reward = self._clip(reward)

        return obs, reward, done, info

    def _clip(self, reward):
        """
            Perform the reward clip.

            Arguments:
                reward    The reward to clip.

            Returns:
                The clipped reward.
        """
        min_value, max_value = self.get_clip_range()
        return np.clip(reward, min_value, max_value)


class AugmentingEnvWrapper(IdentityWrapper):
    """
        This wrapper expects the environment to have a method named
        'augment_observation' that can be utilized to augment an observation
        to create a batch of obserations. Each instance of observation in the
        batch will be coupled with an identical done and reward. This is to
        help a policy learn that a particular augmentation does not affect
        the learned policy.
    """

    def __init__(self,
                 env,
                 **kw_args):
        """
            Initialize the wrapper.

            Arguments:
                env   The environment to wrap.
        """

        super(AugmentingEnvWrapper, self).__init__(
            env,
            **kw_args)

        self.test_idx = -1
        aug_func      = getattr(env, "augment_observation", None)

        if type(aug_func) == type(None):
            msg  = "ERROR: env must define 'augment_observation' in order "
            msg += "to qualify for the AugmentingEnvWrapper class."
            rank_print(msg)
            comm.Abort()

        self.batch_size = self.aug_reset().shape[0]

    def step(self, action):
        """
            Take a single step in the environment using the given
            action. If we're in test mode, we don't augment. Otherwise,
            call the aug_step method.

            NOTE: the action is expected to be a SINGLE action. This does
            not currently support multiple environment instances.

            Arguments:
                action    The action to take.

            Returns:
                The resulting observation(s), reward(s), done(s), and info(s).
        """
        if self.test_mode:
            return self.aug_test_step(action)
        return self.aug_step(action)

    def aug_step(self, action):
        """
            Take a single step in the environment using the given
            action, allow the environment to augment the returned
            observation, and set up the return values as a batch.

            NOTE: the action is expected to be a SINGLE action. This does
            not currently support multiple environment instances.

            Arguments:
                action    The action to take.

            Returns:
                A batch of observations, rewards, and dones along
                with a single info dictionary. The observations will
                contain augmentations of the original observation.
        """
        obs, reward, done, info = self.env.step(action)

        batch_obs  = self.env.augment_observation(obs)
        batch_size = batch_obs.shape[0]

        if "terminal observation" in info[0]:
            batch_infos = np.array([None] * batch_size,
                dtype=object)

            terminal_obs = info[0]["terminal observation"]
            terminal_obs = self.env.augment_observation(terminal_obs)

            for i in range(batch_size):
                i_info = info[0].copy()
                i_info["terminal observation"] = terminal_obs[i].copy()
                batch_infos[i] = i_info.copy()
        else:
            batch_infos = np.tile((info[0],), batch_size)

        batch_rewards = np.tile((reward,), batch_size)
        batch_dones   = np.tile((done,), batch_size).astype(bool)

        batch_rewards = batch_rewards.reshape((batch_size, 1))
        batch_dones   = batch_dones.reshape((batch_size, 1))
        batch_infos   = batch_infos.reshape((batch_size))

        return batch_obs, batch_rewards, batch_dones, batch_infos

    def aug_test_step(self, action):
        """
            Take a single step in the environment using the given
            action, allow the environment to augment the returned
            observation. Since we're in test mode, we return a single
            instance from the batch.

            NOTE: the action is expected to be a SINGLE action. This does
            not currently support multiple environment instances.

            Arguments:
                action    The action to take.

            Returns:
                Observation, reward, done, and info (possibly augmented).
        """
        obs, reward, done, info = self.env.step(action)

        batch_obs  = self.env.augment_observation(obs)
        batch_size = batch_obs.shape[0]

        if self.test_idx < 0:
            self.test_idx = np.random.randint(0, batch_size)

        if "terminal observation" in info:
            terminal_obs = info["terminal observation"]
            terminal_obs = self.env.augment_observation(terminal_obs)
            info["terminal observation"] = terminal_obs[self.test_idx].copy()

        return batch_obs[self.test_idx], reward, done, info

    def reset(self):
        """
            Reset the environment. If we're in test mode, we don't augment the
            resulting observations. Otherwise, augment them before returning.

            Returns:
                The resulting observation(s).
        """
        if self.test_mode:
            return self.aug_test_reset()
        return self.aug_reset()

    def aug_reset(self):
        """
            Reset the environment, and return a batch of augmented observations.

            Returns:
                The resulting observations.
        """
        obs = self.env.reset()

        aug_obs_batch = self.env.augment_observation(obs)

        return aug_obs_batch

    def aug_test_reset(self):
        """
            Reset the environment, and return a single observations which may
            or may not be augmented.

            Returns:
                The resulting observation (possibly augmented).
        """
        obs = self.env.reset()

        aug_obs_batch = self.env.augment_observation(obs)
        batch_size    = aug_obs_batch.shape[0]

        if self.test_idx < 0:
            self.test_idx = np.random.randint(0, batch_size)

        return aug_obs_batch[self.test_idx]

    def get_batch_size(self):
        """
            If any wrapped classes define this method, try to get the batch size
            from them. Otherwise, assume we have a single environment.

            Returns:
                Return our batch size.
        """
        return self.batch_size

    def supports_batched_environments(self):
        """
            Determine whether or not our wrapped environment supports
            batched environments.

            Return:
                True.
        """
        return True

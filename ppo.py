import sys
import pickle
import numpy as np
import os
from copy import deepcopy
import torch
from torch.optim import Adam
from torch import nn
from torch.utils.data import DataLoader
from ppo_and_friends.utils.episode_info import EpisodeInfo, PPODataset
from ppo_and_friends.utils.misc import get_action_dtype, need_action_squeeze
from ppo_and_friends.utils.misc import RunningStatNormalizer
from ppo_and_friends.utils.decrementers import *
from ppo_and_friends.utils.misc import update_optimizer_lr
from ppo_and_friends.networks.icm import ICM
from ppo_and_friends.environments.vectorize import VectorizedEnv
from ppo_and_friends.environments.env_wrappers import ObservationNormalizer, ObservationClipper
from ppo_and_friends.environments.env_wrappers import RewardNormalizer, RewardClipper
from ppo_and_friends.utils.mpi_utils import sync_model_parameters, mpi_avg_gradients
from ppo_and_friends.utils.mpi_utils import mpi_avg
from ppo_and_friends.utils.mpi_utils import rank_print, set_torch_threads
from ppo_and_friends.utils.misc import format_seconds
import time
from gym.spaces import Box, Discrete, MultiDiscrete, MultiBinary
from mpi4py import MPI

comm      = MPI.COMM_WORLD
rank      = comm.Get_rank()
num_procs = comm.Get_size()


class PPO(object):

    def __init__(self,
                 env,
                 ac_network,
                 device,
                 random_seed,
                 icm_network         = ICM,
                 icm_kw_args         = {},
                 actor_kw_args       = {},
                 critic_kw_args      = {},
                 lr                  = 3e-4,
                 min_lr              = 1e-4,
                 lr_dec              = None,
                 max_ts_per_ep       = 64,
                 batch_size          = 256,
                 ts_per_rollout      = 2048,
                 gamma               = 0.99,
                 lambd               = 0.95,
                 epochs_per_iter     = 10,
                 surr_clip           = 0.2,
                 gradient_clip       = 0.5,
                 bootstrap_clip      = (-10.0, 10.0),
                 dynamic_bs_clip     = False,
                 use_gae             = True,
                 use_icm             = False,
                 icm_beta            = 0.8,
                 ext_reward_weight   = 1.0,
                 intr_reward_weight  = 1.0,
                 entropy_weight      = 0.01,
                 target_kl           = 0.015,
                 mean_window_size    = 100,
                 normalize_adv       = True,
                 normalize_obs       = False,
                 normalize_rewards   = False,
                 normalize_values    = False,
                 obs_clip            = None,
                 reward_clip         = None,
                 render              = False,
                 load_state          = False,
                 state_path          = "./",
                 save_best_only      = False,
                 pickle_class        = False,
                 use_soft_resets     = True,
                 test_mode           = False):
        """
            Initialize the PPO trainer.

            Parameters:
                 env                  The environment to learn from.
                 ac_network           The actor/critic network.
                 device               A torch device to use for training.
                 random_seed          A random seed to use.
                 icm_network          The network to use for ICM applications.
                 icm_kw_args          Extra keyword args for the ICM network.
                 actor_kw_args        Extra keyword args for the actor
                                      network.
                 critic_kw_args       Extra keyword args for the critic
                                      network.
                 lr                   The initial learning rate.
                 min_lr               The minimum learning rate.
                 lr_dec               A class that inherits from the Decrement
                                      class located in utils/decrementers.py.
                                      This class has a decrement function that
                                      will be used to updated the learning rate.
                 max_ts_per_ep        The maximum timesteps to allow per
                                      episode.
                 batch_size           The batch size to use when training/
                                      updating the networks.
                 ts_per_rollout       A soft limit on the number of timesteps
                                      to allow per rollout (can span multiple
                                      episodes). Note that our actual timestep
                                      count can exceed this limit, but we won't
                                      start any new episodes once it has.
                 gamma                The 'gamma' value for calculating
                                      advantages and discounting rewards
                                      when normalizing them.
                 lambd                The 'lambda' value for calculating GAEs.
                 epochs_per_iter      'Epoch' is used loosely and with a variety
                                      of meanings in RL. In this case, a single
                                      epoch is a single update of all networks.
                                      epochs_per_iter is the number of updates
                                      to perform after a single rollout (which
                                      may contain multiple episodes).
                 surr_clip            The clip value applied to the surrogate
                                      (standard PPO approach).
                 gradient_clip        A clip value to use on the gradient
                                      update.
                 bootstrap_clip       When using GAE, we bootstrap the values
                                      and rewards when an epsiode is cut off
                                      before completion. In these cases, we
                                      clip the bootstrapped reward to a
                                      specific range. Why is this? Well, our
                                      estimated reward (from our value network)
                                      might be way outside of the expected
                                      range.
                 dynamic_bs_clip      If set to True, bootstrap_clip will be
                                      used as the initial clip values, but all
                                      values thereafter will be taken from the
                                      global min and max rewards that have been
                                      seen so far.
                 use_gae              Should we use Generalized Advantage
                                      Estimations? If not, fall back on the
                                      vanilla advantage calculation.
                 use_icm              Should we use an Intrinsic Curiosity
                                      Module?
                 icm_beta             The beta value used within the ICM.
                 ext_reward_weight    An optional weight for the extrinsic
                                      reward.
                 intr_reward_weight   an optional weight for the intrinsic
                                      reward.
                 entropy_weight       An optional weight to apply to our
                                      entropy.
                 target_kl            KL divergence used for early stopping.
                                      This is typically set in the range
                                      [0.1, 0.5]. Use high values to disable.
                 mean_window_size     The window size for a running mean. Note
                                      that each "score" in the window is
                                      actually the mean score for that rollout.
                 normalize_adv        Should we normalize the advantages? This
                                      occurs at the minibatch level.
                 normalize_obs        Should we normalize the observations?
                 normalize_rewards    Should we normalize the rewards?
                 normalize_values     Should we normalize the "values" that our
                                      critic calculates loss against?
                 obs_clip             Disabled if None. Otherwise, this should
                                      be a tuple containing a clip range for
                                      the observation space as (min, max).
                 reward_clip          Disabled if None. Otherwise, this should
                                      be a tuple containing a clip range for
                                      the reward as (min, max).
                 render               Should we render the environment while
                                      training?
                 load_state           Should we load a saved state?
                 state_path           The path to save/load our state.
                 save_best_only       When enabled, only save the models when
                                      the top mean window increases. Note that
                                      this assumes that the top scores will be
                                      the best policy, which might not always
                                      hold up, but it's a general assumption.
                 pickle_class         When enabled, the entire PPO class will
                                      be pickled and saved into the output
                                      directory after it's been initialized.
                 use_soft_resets      Use "soft resets" during rollouts.
                 test_mode            Most of this class is not used for
                                      testing, but some of its attributes are.
                                      Setting this to True will enable test
                                      mode.
        """
        set_torch_threads()

        #
        # Divide the ts per rollout up among the processors. Let rank
        # 0 take any excess.
        #
        orig_ts        = ts_per_rollout
        ts_per_rollout = int(ts_per_rollout / num_procs)
        if rank == 0:
            ts_per_rollout += orig_ts % num_procs

        if not test_mode:
            rank_print("ts_per_rollout per rank: ~{}".format(ts_per_rollout))

        #
        # For reproducibility, we need to set the environment's random
        # seeds. Let's allow testing to be random.
        #
        if not test_mode:
            env.action_space.seed(random_seed)
            env.seed(random_seed)

        #
        # Vectorize our environment and add any requested wrappers.
        #
        env = VectorizedEnv(env)

        self.save_env_info = False

        #
        # Let's wrap our environment in the requested wrappers. Note that
        # order matters!
        #
        if normalize_obs:
            env = ObservationNormalizer(
                env          = env,
                test_mode    = test_mode,
                update_stats = not test_mode)

            self.save_env_info = True

        if obs_clip != None and type(obs_clip) == tuple:
            env = ObservationClipper(
                env        = env,
                test_mode  = test_mode,
                clip_range = obs_clip)

        #
        # There are multiple ways to go about normalizing values/rewards.
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

            self.save_env_info = True

        if reward_clip != None and type(reward_clip) == tuple:
            env = RewardClipper(
                env        = env,
                test_mode  = test_mode,
                clip_range = reward_clip)

        #
        # When we toggle test mode on/off, we need to make sure to also
        # toggle this flag for any modules that depend on it.
        #
        self.test_mode_dependencies = [env]

        act_type = type(env.action_space)

        if (issubclass(act_type, Box) or
            issubclass(act_type, MultiBinary) or
            issubclass(act_type, MultiDiscrete)):

            self.act_dim = env.action_space.shape

        elif issubclass(act_type, Discrete):
            self.act_dim = env.action_space.n

        else:
            msg = "ERROR: unsupported action space {}".format(env.action_space)
            rank_print(msg)
            comm.Abort()

        if (issubclass(act_type, MultiBinary) or
            issubclass(act_type, MultiDiscrete)):
            msg  = "WARNING: MultiBinary and MultiDiscrete action spaces "
            msg += "may not be fully supported. Use at your own risk."
            rank_print(msg)

        action_dtype = get_action_dtype(env)

        if action_dtype == "unknown":
            rank_print("ERROR: unknown action type!")
            comm.Abort()
        else:
            rank_print("Using {} actions.".format(action_dtype))

        #
        # Environments are very inconsistent! We need to check what shape
        # they expect actions to be in.
        #
        self.action_squeeze = need_action_squeeze(env)

        if lr_dec == None:
            self.lr_dec = LinearDecrementer(
                max_iteration = 2000,
                max_value     = lr,
                min_value     = min_lr)
        else:
            self.lr_dec = lr_dec

        self.env                 = env
        self.device              = device
        self.state_path          = state_path
        self.render              = render
        self.action_dtype        = action_dtype
        self.use_gae             = use_gae
        self.use_icm             = use_icm
        self.icm_beta            = icm_beta
        self.ext_reward_weight   = ext_reward_weight
        self.intr_reward_weight  = intr_reward_weight
        self.min_lr              = min_lr
        self.max_ts_per_ep       = max_ts_per_ep
        self.batch_size          = batch_size
        self.ts_per_rollout      = ts_per_rollout
        self.gamma               = gamma
        self.lambd               = lambd
        self.target_kl           = target_kl
        self.epochs_per_iter     = epochs_per_iter
        self.surr_clip           = surr_clip
        self.gradient_clip       = gradient_clip
        self.bootstrap_clip      = bootstrap_clip
        self.dynamic_bs_clip     = dynamic_bs_clip
        self.entropy_weight      = entropy_weight
        self.obs_shape           = env.observation_space.shape
        self.prev_top_window     = -np.finfo(np.float32).max
        self.save_best_only      = save_best_only
        self.mean_window_size    = mean_window_size 
        self.normalize_adv       = normalize_adv
        self.normalize_rewards   = normalize_rewards
        self.normalize_values    = normalize_values
        self.score_cache         = np.zeros(0)
        self.lr                  = lr
        self.use_soft_resets     = use_soft_resets
        self.test_mode           = test_mode

        #
        # Create a dictionary to track the status of training.
        #
        max_int           = np.iinfo(np.int32).max
        self.status_dict  = {}
        self.status_dict["iteration"]            = 0
        self.status_dict["rollout time"]         = 0
        self.status_dict["running time"]         = 0
        self.status_dict["timesteps"]            = 0
        self.status_dict["longest run"]          = 0
        self.status_dict["window avg"]           = 'N/A'
        self.status_dict["top window avg"]       = 'N/A'
        self.status_dict["episode reward avg"]   = 0
        self.status_dict["extrinsic score avg"]  = 0
        self.status_dict["top score"]            = -max_int
        self.status_dict["total episodes"]       = 0
        self.status_dict["weighted entropy"]     = 0
        self.status_dict["actor loss"]           = 0
        self.status_dict["critic loss"]          = 0
        self.status_dict["kl avg"]               = 0
        self.status_dict["lr"]                   = self.lr
        self.status_dict["reward range"]         = (max_int, -max_int)
        self.status_dict["obs range"]            = (max_int, -max_int)

        #
        # Value normalization is discussed in multiple papers, so I'm not
        # going to reference one in particular. In general, the idea is
        # to normalize the targets of the critic network using a running
        # average. The output of the critic then needs to be de-normalized
        # for calcultaing advantages.
        #
        if normalize_values:
            self.value_normalizer = RunningStatNormalizer(
                name      = "value_normalizer",
                device    = self.device,
                test_mode = test_mode)

            self.test_mode_dependencies.append(self.value_normalizer)

        if save_best_only:
            self.status_dict["last save"] = -1

        #
        # Some methods (ICM) perform best if we can clone the environment,
        # but not all environments support this.
        #
        try:
            self.env.reset()
            cloned_env = deepcopy(self.env)
            cloned_env.step(cloned_env.action_space.sample())
            self.can_clone_env = True
        except:
            self.can_clone_env = False

        #
        # Initialize our networks: actor, critic, and possible ICM.
        #
        use_conv2d_setup = False
        for base in ac_network.__bases__:
            if base.__name__ == "PPOConv2dNetwork":
                use_conv2d_setup = True

        #
        # arXiv:2006.05990v1 suggests initializing the output layer
        # of the actor network with a weight that's ~100x smaller
        # than the rest of the layers. We initialize layers with a
        # value near 1.0 by default, so we set the last layer to
        # 0.01. The same paper also suggests that the last layer of
        # the value network doesn't matter so much. I can't remember
        # where I got 1.0 from... I'll try to track that down.
        #
        if use_conv2d_setup:
            obs_dim = self.obs_shape

            self.actor = ac_network(
                name         = "actor", 
                in_shape     = obs_dim,
                out_dim      = self.act_dim, 
                out_init     = 0.01,
                action_dtype = action_dtype,
                test_mode    = test_mode,
                **actor_kw_args)

            self.critic = ac_network(
                name         = "critic", 
                in_shape     = obs_dim,
                out_dim      = 1,
                out_init     = 1.0,
                action_dtype = action_dtype,
                test_mode    = test_mode,
                **critic_kw_args)

        else:
            obs_dim = self.obs_shape[0]

            self.actor = ac_network(
                name         = "actor", 
                in_dim       = obs_dim, 
                out_dim      = self.act_dim, 
                out_init     = 0.01,
                action_dtype = action_dtype,
                test_mode    = test_mode,
                **actor_kw_args)

            self.critic = ac_network(
                name         = "critic", 
                in_dim       = obs_dim, 
                out_dim      = 1,
                out_init     = 1.0,
                action_dtype = action_dtype,
                test_mode    = test_mode,
                **critic_kw_args)

        self.actor  = self.actor.to(device)
        self.critic = self.critic.to(device)

        self.test_mode_dependencies.append(self.actor)
        self.test_mode_dependencies.append(self.critic)

        sync_model_parameters(self.actor)
        sync_model_parameters(self.critic)
        comm.barrier()

        if self.use_icm:
            self.icm_model = icm_network(
                name         = "icm",
                obs_dim      = obs_dim,
                act_dim      = self.act_dim,
                action_dtype = self.action_dtype,
                test_mode    = test_mode,
                **icm_kw_args)

            self.test_mode_dependencies.append(self.icm)

            self.icm_model.to(device)
            self.status_dict["icm loss"] = 0
            self.status_dict["intrinsic score avg"] = 0

            sync_model_parameters(self.icm_model)
            comm.barrier()

        if load_state:
            if not os.path.exists(state_path):
                msg  = "WARNING: state_path does not exist. Unable "
                msg += "to load state."
                rank_print(msg)
            else:
                #
                # Let's ensure backwards compatibility with previous commits.
                #
                tmp_status_dict = self.load()

                for key in tmp_status_dict:
                    if key in self.status_dict:
                        self.status_dict[key] = tmp_status_dict[key]

                self.lr= min(self.status_dict["lr"], self.lr)
                self.status_dict["lr"] = self.lr

        self.actor_optim  = Adam(self.actor.parameters(), lr=lr, eps=1e-5)
        self.critic_optim = Adam(self.critic.parameters(), lr=lr, eps=1e-5)

        if self.use_icm:
            self.icm_optim = Adam(self.icm_model.parameters(),
                lr=lr, eps=1e-5)

        if not os.path.exists(state_path) and rank == 0:
            os.makedirs(state_path)
        comm.barrier()

        #
        # If requested, pickle the entire class. This is useful for situations
        # where we want to load a trained model into a particular env for
        # testing or deploying.
        #
        if pickle_class and rank == 0:
            file_name  = "PPO.pickle"
            state_file = os.path.join(self.state_path, file_name)
            with open(state_file, "wb") as out_f:
                pickle.dump(self, out_f,
                    protocol=pickle.HIGHEST_PROTOCOL)

        comm.barrier()


    def get_action(self, obs):
        """
            Given an observation from our environment, determine what the
            action should be.

            Arguments:
                obs    The environment observation.

            Returns:
                A tuple of form (raw_action, action, log_prob) s.t. "raw_action"
                is the distribution sample before any "squashing" takes place,
                "action" is the the action value that should be fed to the
                environment, and log_prob is the log probabilities from our
                probability distribution.
        """
        t_obs = torch.tensor(obs, dtype=torch.float).to(self.device)
        t_obs = t_obs.unsqueeze(0)

        with torch.no_grad():
            action_pred = self.actor(t_obs)

        action_pred = action_pred.cpu().detach()
        dist        = self.actor.distribution.get_distribution(action_pred)

        #
        # Our distribution gives us two potentially distinct actions, one of
        # which is guaranteed to be a raw sample from the distribution. The
        # other might be altered in some way (usually to enforce a range).
        #
        action, raw_action = self.actor.distribution.sample_distribution(dist)
        log_prob = self.actor.distribution.get_log_probs(dist, raw_action)

        if self.action_dtype == "discrete":
            action = action.int().unsqueeze(0)

        action     = action.detach().numpy()
        raw_action = raw_action.detach().numpy()

        return raw_action, action, log_prob.detach()

    def evaluate(self, batch_obs, batch_actions):
        """
            Given a batch of observations, use our critic to approximate
            the expected return values. Also use a batch of corresponding
            actions to retrieve some other useful information.

            Arguments:
                batch_obs      A batch of observations.
                batch_actions  A batch of actions corresponding to the batch of
                               observations.

            Returns:
                A tuple of form (values, log_probs, entropies) s.t. values are
                the critic predicted value, log_probs are the log probabilities
                from our probability distribution, and entropies are the
                entropies from our distribution.
        """
        values = self.critic(batch_obs).squeeze()

        action_pred = self.actor(batch_obs).cpu()
        dist        = self.actor.distribution.get_distribution(action_pred)

        if self.action_dtype == "continuous" and len(batch_actions.shape) < 2:
            log_probs = self.actor.distribution.get_log_probs(
                dist,
                batch_actions.unsqueeze(1).cpu())
        else:
            log_probs = self.actor.distribution.get_log_probs(
                dist,
                batch_actions.cpu())

        entropy = self.actor.distribution.get_entropy(dist, action_pred)

        return values, log_probs.to(self.device), entropy.to(self.device)

    def print_status(self):
        """
            Print out statistics from our status_dict.
        """
        rank_print("\n--------------------------------------------------------")
        rank_print("Status Report:")
        for key in self.status_dict:

            if key == "running time" or key == "rollout time":
                pretty_time = format_seconds(self.status_dict[key])
                rank_print("    {}: {}".format(key, pretty_time))
            else:
                rank_print("    {}: {}".format(key, self.status_dict[key]))

        rank_print("--------------------------------------------------------")

    def update_learning_rate(self,
                             iteration):
        """
            Update the learning rate. This relies on the rl_dec function,
            which expects an iteration and returns an updated learning rate.

            Arguments:
                iteration    The current iteration of training.
        """

        self.lr = self.lr_dec(iteration)

        update_optimizer_lr(self.actor_optim, self.lr)
        update_optimizer_lr(self.critic_optim, self.lr)

        self.lr = self.lr_dec(iteration)

        update_optimizer_lr(self.actor_optim, self.lr)
        update_optimizer_lr(self.critic_optim, self.lr)

        if self.use_icm:
            update_optimizer_lr(self.icm_optim, self.lr)

        self.status_dict["lr"] = self.actor_optim.param_groups[0]["lr"]

    def get_intrinsic_reward(self,
                             prev_obs,
                             obs,
                             action):
        """
            Query the ICM for an intrinsic reward.

            Arguments:
                prev_obs    The previous observation (before the latest
                            action).
                obs         The current observation.
                action      The action taken.
        """

        obs_1 = torch.tensor(prev_obs,
            dtype=torch.float).to(self.device).unsqueeze(0)
        obs_2 = torch.tensor(obs,
            dtype=torch.float).to(self.device).unsqueeze(0)

        if self.action_dtype == "discrete":
            action = torch.tensor(action,
                dtype=torch.long).to(self.device).unsqueeze(0)

        elif self.action_dtype == "continuous":
            action = torch.tensor(action,
                dtype=torch.float).to(self.device)

        if len(action.shape) != 2:
            action = action.unsqueeze(0)

        with torch.no_grad():
            intr_reward, _, _ = self.icm_model(obs_1, obs_2, action)

        intr_reward  = intr_reward.item()
        intr_reward *= self.intr_reward_weight

        return intr_reward


    def rollout(self):
        """
            Create a "rollout" of episodes. This system uses "fixed-length
            trajectories", which are sometimes referred to as "vectorized"
            episodes. In short, we step through our environment for a fixed
            number of iterations, and only allow a fixed number of steps
            per episode. This fixed number of steps per episode becomes a
            trajectory. In most cases, our trajectory length < max steps
            in the environment, which results in trajectories ending before
            the episode ends. In those cases, we bootstrap the ending value
            by using our critic to approximate the next value. A key peice
            of this logic is that the enviorment's state is saved after a
            trajectory ends, meaning that a new trajectory can start in the
            middle of an episode.

            Returns:
                A PyTorch dataset containing our rollout.
        """
        if self.env ==  None:
            msg  = "ERROR: unable to perform rollout due to the environment "
            msg += "being of type None. This is likey due to loading the "
            msg += "PPO class from a pickled state."
            rank_print(msg)
            comm.Abort()

        dataset            = PPODataset(self.device, self.action_dtype)
        total_episodes     = 0.0
        total_rollout_ts   = 0
        total_ext_rewards  = 0
        total_rewards      = 0
        total_intr_rewards = 0
        top_rollout_score  = -np.finfo(np.float32).max
        ep_rewards         = 0
        longest_run        = 0
        rollout_max_reward = -np.finfo(np.float32).max
        rollout_min_reward = np.finfo(np.float32).max
        rollout_max_obs    = -np.finfo(np.float32).max
        rollout_min_obs    = np.finfo(np.float32).max
        episode_length     = 0
        ep_score           = 0

        self.actor.eval()
        self.critic.eval()

        if self.use_icm:
            self.icm_model.eval()

        #
        # TODO: soft resets might cause rollouts to start off in "traps"
        # that are impossible to escape. We might be able to handle this
        # more intelligently.
        #
        if self.use_soft_resets:
            obs = self.env.soft_reset()
        else:
            obs = self.env.reset()

        start_time = time.time()

        while total_rollout_ts < self.ts_per_rollout:

            episode_info = EpisodeInfo(
                use_gae        = self.use_gae,
                gamma          = self.gamma,
                lambd          = self.lambd,
                bootstrap_clip = self.bootstrap_clip)

            for ep_ts in range(1, self.max_ts_per_ep + 1):

                #
                # We end if we've reached our timesteps per rollout limit.
                #
                if ep_ts > self.ts_per_rollout:
                    msg  = "ERROR: the episode timestep is > timesteps "
                    msg += "per rollout. This is not allowable."
                    msg += "episode timestep, max timesteps: {}, ".format(ep_ts)
                    msg += "{}.".format(self.ts_per_rollout)
                    rank_print(msg)
                    comm.Abort()

                if self.render:
                    self.env.render()

                total_rollout_ts += 1
                episode_length   += 1
                raw_action, action, log_prob = self.get_action(obs)

                t_obs = torch.tensor(obs, dtype=torch.float).to(self.device)
                t_obs = t_obs.unsqueeze(0)
                value = self.critic(t_obs)

                if self.normalize_values:
                    value = self.value_normalizer.denormalize(value)

                prev_obs = obs.copy()

                if self.action_squeeze:
                    action = action.squeeze()

                obs, ext_reward, done, info = self.env.step(action)

                if action.size == 1:
                    ep_action  = action.item()
                    raw_action = raw_action.item()
                else:
                    ep_action = action

                ep_value = value.item()

                #
                # If any of our wrappers are altering the rewards, there should
                # be an unaltered version in the info.
                #
                if "natural reward" in info:
                    natural_reward = info["natural reward"]
                else:
                    natural_reward = ext_reward

                ext_reward *= self.ext_reward_weight

                #
                # If we're using the ICM, we need to do some extra work here.
                # This amounts to adding "curiosity", aka intrinsic reward,
                # to out extrinsic reward.
                #
                if self.use_icm:

                    intr_reward = self.get_intrinsic_reward(
                        prev_obs,
                        obs,
                        action)

                    total_intr_rewards += intr_reward
                    reward = ext_reward + intr_reward

                else:
                    reward = ext_reward

                ep_obs = obs
                if done:
                    ep_obs = info["terminal observation"]

                episode_info.add_info(
                    prev_obs,
                    ep_obs,
                    raw_action,
                    ep_action,
                    ep_value,
                    log_prob,
                    reward)

                rollout_max_reward = max(rollout_max_reward, reward)
                rollout_min_reward = min(rollout_min_reward, reward)
                rollout_max_obs    = max(rollout_max_obs, obs.max())
                rollout_min_obs    = min(rollout_min_obs, obs.min())

                ep_rewards += reward
                ep_score   += natural_reward

                if done:
                    episode_info.end_episode(
                        ending_value   = 0,
                        ending_reward  = 0,
                        episode_length = episode_length)

                    dataset.add_episode(episode_info)

                    if self.dynamic_bs_clip:
                        ep_min_reward       = min(episode_info.rewards)
                        ep_max_reward       = max(episode_info.rewards)
                        self.bootstrap_clip = (ep_min_reward, ep_max_reward)

                    episode_info = EpisodeInfo(
                        use_gae        = self.use_gae,
                        gamma          = self.gamma,
                        lambd          = self.lambd,
                        bootstrap_clip = self.bootstrap_clip)

                    longest_run = max(longest_run, episode_length)

                    total_ext_rewards += ep_score
                    total_rewards     += ep_rewards
                    top_rollout_score  = max(top_rollout_score, ep_score)
                    episode_length     = 0
                    ep_score           = 0
                    ep_rewards         = 0
                    total_episodes    += 1.

                elif (ep_ts == self.max_ts_per_ep or
                    total_rollout_ts == self.ts_per_rollout):

                    t_obs      = torch.tensor(obs, dtype=torch.float).to(self.device)
                    t_obs      = t_obs.unsqueeze(0)
                    nxt_value  = self.critic(t_obs)

                    if self.normalize_values:
                        nxt_value = self.value_normalizer.denormalize(nxt_value)

                    nxt_value  = nxt_value.item()
                    nxt_reward = nxt_value

                    #
                    # Tricky business:
                    # Typically, we just use the result of our critic to
                    # bootstrap the expected reward. This is problematic
                    # with ICM because we can't really expect our critic to
                    # learn about "surprise". I dont' know of any perfect
                    # ways to handle this, but here are some ideas:
                    #
                    #     1. Just use the value anyways. As long as the
                    #        max ts per episode is long enough, we'll
                    #        hopefully see enough intrinsic reward to
                    #        learn a good policy. In my experience, this
                    #        works, but it takes a bit longer to learn.
                    #     2. If we can clone the environment, we can take
                    #        an extra step with the clone to get the
                    #        intrinsic reward, and we can decide what to
                    #        do with this. Approaches that integrate this
                    #        reward tend to learn a bit more quickly.
                    #
                    # If we have this intrinsic reward from a clone step,
                    # we can hand wavily calcluate a "surprise" by taking
                    # the difference between the average intrinsic reward
                    # and the one we get. Adding that to the critic's
                    # output can act as an extra surprise bonus.
                    #
                    if self.use_icm:
                        if self.can_clone_env:
                            _, clone_action, _ = self.get_action(obs)

                            if self.action_squeeze:
                                clone_action = clone_action.squeeze()

                            clone_prev_obs = obs.copy()
                            cloned_env = deepcopy(self.env)
                            clone_obs, _, _, _ = cloned_env.step(clone_action)
                            del cloned_env

                            intr_reward = self.get_intrinsic_reward(
                                clone_prev_obs,
                                clone_obs,
                                clone_action)

                        ism         = self.status_dict["intrinsic score avg"]
                        surprise    = intr_reward - ism
                        nxt_reward += surprise

                    episode_info.end_episode(
                        ending_value   = nxt_value,
                        ending_reward  = nxt_reward,
                        episode_length = episode_length)

                    dataset.add_episode(episode_info)

                    if self.dynamic_bs_clip:
                        ep_min_reward       = min(episode_info.rewards)
                        ep_max_reward       = max(episode_info.rewards)
                        self.bootstrap_clip = (ep_min_reward, ep_max_reward)

                    episode_info = EpisodeInfo(
                        use_gae        = self.use_gae,
                        gamma          = self.gamma,
                        lambd          = self.lambd,
                        bootstrap_clip = self.bootstrap_clip)

                    if total_rollout_ts == self.ts_per_rollout:
                        ts_before_ep  = self.ts_per_rollout - episode_length
                        current_total = total_episodes

                        if current_total == 0:
                            current_total = 1.0

                        if ts_before_ep == 0:
                            avg_ep_len = self.ts_per_rollout
                        else:
                            avg_ep_len = ts_before_ep / current_total

                        ep_perc            = episode_length / avg_ep_len
                        total_episodes    += ep_perc
                        total_ext_rewards += ep_score
                        total_rewards     += ep_rewards

            longest_run = max(longest_run, episode_length)

        if total_episodes <= 1.0:
            top_rollout_score = max(top_rollout_score, ep_score)

        #
        # Update our status dict.
        #
        top_score = max(top_rollout_score, self.status_dict["top score"])
        top_score = comm.allreduce(top_score, MPI.MAX)

        rollout_max_reward = max(self.status_dict["reward range"][1],
            rollout_max_reward)
        rollout_max_reward = comm.allreduce(rollout_max_reward, MPI.MAX)

        rollout_min_reward = min(self.status_dict["reward range"][0],
            rollout_min_reward)
        rollout_min_reward = comm.allreduce(rollout_min_reward, MPI.MIN)

        rollout_max_obs = max(self.status_dict["obs range"][1],
            rollout_max_obs)
        rollout_max_obs = comm.allreduce(rollout_max_obs, MPI.MAX)

        rollout_min_obs = min(self.status_dict["obs range"][0],
            rollout_min_obs)
        rollout_min_obs = comm.allreduce(rollout_min_obs, MPI.MIN)

        longest_run       = comm.allreduce(longest_run, MPI.MAX)
        total_episodes    = comm.allreduce(total_episodes, MPI.SUM)
        total_ext_rewards = comm.allreduce(total_ext_rewards, MPI.SUM)
        total_rewards     = comm.allreduce(total_rewards, MPI.SUM)
        total_rollout_ts  = comm.allreduce(total_rollout_ts, MPI.SUM)

        running_ext_score = total_ext_rewards / total_episodes
        running_score     = total_rewards / total_episodes
        rw_range          = (rollout_min_reward, rollout_max_reward)
        obs_range         = (rollout_min_obs, rollout_max_obs)

        self.status_dict["episode reward avg"]  = running_score
        self.status_dict["extrinsic score avg"] = running_ext_score
        self.status_dict["top score"]           = top_score
        self.status_dict["total episodes"]     += total_episodes
        self.status_dict["longest run"]         = longest_run
        self.status_dict["reward range"]        = rw_range
        self.status_dict["obs range"]           = obs_range
        self.status_dict["timesteps"]          += total_rollout_ts

        #
        # Update our score cache and the window mean when appropriate.
        #
        if self.score_cache.size < self.mean_window_size:
            self.score_cache = np.append(self.score_cache, running_score)
            self.status_dict["window avg"]  = "N/A"
        else:
            self.score_cache = np.roll(self.score_cache, -1)
            self.score_cache[-1] = running_score

            self.status_dict["window avg"]  = self.score_cache.mean()

            if type(self.status_dict["top window avg"]) == str:
                self.status_dict["top window avg"] = \
                    self.status_dict["window avg"]

            elif (self.status_dict["window avg"] >
                self.status_dict["top window avg"]):

                self.status_dict["top window avg"] = \
                    self.status_dict["window avg"]

        if self.use_icm:
            total_intr_rewards = comm.allreduce(total_intr_rewards, MPI.SUM)
            ism = total_intr_rewards / total_episodes
            self.status_dict["intrinsic score avg"] = ism

        #
        # Finally, bulid the dataset.
        #
        dataset.build()

        comm.barrier()
        stop_time = time.time()
        self.status_dict["rollout time"] = stop_time - start_time

        return dataset

    def learn(self, num_timesteps):
        """
            Learn!
                1. Create a rollout dataset.
                2. Update our networks.
                3. Repeat until we've reached our max timesteps.

            Arguments:
                num_timesteps    The maximum number of timesteps to run.
                                 Note that this is in addtion to however
                                 many timesteps were run during the last save.
        """

        start_time = time.time()
        ts_max     = self.status_dict["timesteps"] + num_timesteps

        while self.status_dict["timesteps"] < ts_max:
            iter_start_time = time.time()

            dataset = self.rollout()

            self.status_dict["iteration"] += 1

            self.update_learning_rate(self.status_dict["iteration"])

            data_loader = DataLoader(
                dataset,
                batch_size = self.batch_size,
                shuffle    = True)

            self.actor.train()
            self.critic.train()

            if self.use_icm:
                self.icm_model.train()

            # TODO: there is some evidence that re-computing advantages at
            # each epoch can improve training performance. Let's try this
            # out.
            for i in range(self.epochs_per_iter):

                self._ppo_batch_train(data_loader)

                #
                # Early ending using KL. Why multiply by 1.5, you ask? I have
                # no idea, really. It's a magic number that the folks at
                # OpenAI are using.
                #
                comm.barrier()
                if self.status_dict["kl avg"] > (1.5 * self.target_kl):
                    msg  = "\nTarget KL of {} ".format(1.5 * self.target_kl)
                    msg += "has been reached. "
                    msg += "Ending early (after {} epochs)".format(i + 1)
                    rank_print(msg)
                    break

            if self.use_icm:
                for _ in range(self.epochs_per_iter):
                    self._icm_batch_train(data_loader)


            running_time = (time.time() - iter_start_time)
            self.status_dict["running time"] += running_time
            self.print_status()

            need_save = False
            if type(self.status_dict["top window avg"]) == str:
                need_save = True
            elif (self.save_best_only and
                self.status_dict["window avg"] ==
                self.status_dict["top window avg"]):
                need_save = True
            elif not self.save_best_only:
                need_save = True

            if need_save:
                self.save()

                if "last save" in self.status_dict:
                    self.status_dict["last save"] = \
                        self.status_dict["iteration"]

            comm.barrier()
            if self.lr <= 0.0:
                rank_print("Learning rate has bottomed out. Terminating early")
                break

        stop_time   = time.time()
        seconds     = (stop_time - start_time)
        pretty_time = format_seconds(seconds)
        rank_print("Time spent training: {}".format(pretty_time))

    def _ppo_batch_train(self, data_loader):
        """
            Train our PPO networks using mini batches.

            Arguments:
                data_loader    A PyTorch data loader.
        """
        total_actor_loss  = 0
        total_critic_loss = 0
        total_entropy     = 0
        total_w_entropy   = 0
        total_kl          = 0
        counter           = 0

        for obs, _, raw_actions, _, advantages, log_probs, rewards_tg \
            in data_loader:

            torch.cuda.empty_cache()

            if self.normalize_values:
                rewards_tg = self.value_normalizer.normalize(rewards_tg)

            if obs.shape[0] == 1:
                rank_print("Skipping batch of size 1")
                rank_print("    obs shape: {}".format(obs.shape))
                continue

            #
            # arXiv:2005.12729v1 suggests that normalizing advantages
            # at the mini-batch level increases performance.
            #
            if self.normalize_adv:
                adv_std  = advantages.std()
                adv_mean = advantages.mean()
                if torch.isnan(adv_std):
                    rank_print("\nAdvantages std is nan!")
                    rank_print("Advantages:\n{}".format(advantages))
                    comm.Abort()

                advantages = (advantages - adv_mean) / (adv_std + 1e-8)

            values, curr_log_probs, entropy = self.evaluate(obs, raw_actions)

            #
            # The heart of PPO: arXiv:1707.06347v2
            #
            ratios = torch.exp(curr_log_probs - log_probs)
            surr1  = ratios * advantages
            surr2  = torch.clamp(
                ratios, 1 - self.surr_clip, 1 + self.surr_clip) * advantages

            total_kl += (log_probs - curr_log_probs).mean().item()

            if torch.isnan(ratios).any() or torch.isinf(ratios).any():
                rank_print("ERROR: ratios are nan or inf!")

                ratios_min = ratios.min()
                ratios_max = ratios.max()
                rank_print("ratios min, max: {}, {}".format(
                    ratios_min, ratios_max))

                clp_min = curr_log_probs.min()
                clp_max = curr_log_probs.min()
                rank_print("curr_log_probs min, max: {}, {}".format(
                    clp_min, clp_max))

                lp_min = log_probs.min()
                lp_max = log_probs.min()
                rank_print("log_probs min, max: {}, {}".format(
                    lp_min, lp_max))

                act_min = raw_actions.min()
                act_max = raw_actions.max()
                rank_print("actions min, max: {}, {}".format(
                    act_min, act_max))

                std = nn.functional.softplus(self.actor.distribution.log_std)
                rank_print("actor std: {}".format(std))

                comm.Abort()

            #
            # We negate here to perform gradient ascent rather than descent.
            #
            actor_loss        = (-torch.min(surr1, surr2)).mean()
            total_actor_loss += actor_loss.item()

            if self.entropy_weight != 0.0:
                total_entropy += entropy.mean().item()
                actor_loss    -= self.entropy_weight * entropy.mean()

            if values.size() == torch.Size([]):
                values = values.unsqueeze(0)

            # TODO: should we add an option for value clipping? Research
            # suggests this might not be that beneficial, but it might
            # be nice to have it as an option...
            critic_loss        = nn.MSELoss()(values, rewards_tg)
            total_critic_loss += critic_loss.item()

            #
            # arXiv:2005.12729v1 suggests that gradient clipping can
            # have a positive effect on training.
            #
            self.actor_optim.zero_grad()
            actor_loss.backward()
            mpi_avg_gradients(self.actor)
            nn.utils.clip_grad_norm_(self.actor.parameters(),
                self.gradient_clip)
            self.actor_optim.step()

            self.critic_optim.zero_grad()
            critic_loss.backward()
            mpi_avg_gradients(self.critic)
            nn.utils.clip_grad_norm_(self.critic.parameters(),
                self.gradient_clip)
            self.critic_optim.step()

            comm.barrier()
            counter += 1

        counter           = comm.allreduce(counter, MPI.SUM)
        total_entropy     = comm.allreduce(total_entropy, MPI.SUM)
        total_actor_loss  = comm.allreduce(total_actor_loss, MPI.SUM)
        total_critic_loss = comm.allreduce(total_critic_loss, MPI.SUM)
        total_kl          = comm.allreduce(total_kl, MPI.SUM)
        w_entropy         = total_entropy * self.entropy_weight

        self.status_dict["weighted entropy"] = w_entropy / counter
        self.status_dict["actor loss"]       = total_actor_loss / counter
        self.status_dict["critic loss"]      = total_critic_loss / counter
        self.status_dict["kl avg"]           = total_kl / counter

    def _icm_batch_train(self, data_loader):
        """
            Train our ICM networks using mini batches.

            Arguments:
                data_loader    A PyTorch data loader.
        """

        total_icm_loss = 0
        counter = 0

        for obs, next_obs, _, actions, _, _, _ in data_loader:
            torch.cuda.empty_cache()

            actions = actions.unsqueeze(1)

            _, inv_loss, f_loss = self.icm_model(obs, next_obs, actions)

            icm_loss = (((1.0 - self.icm_beta) * f_loss) +
                (self.icm_beta * inv_loss))

            total_icm_loss += icm_loss.item()

            self.icm_optim.zero_grad()
            icm_loss.backward()
            mpi_avg_gradients(self.icm_model)
            self.icm_optim.step()

            counter += 1
            comm.barrier()

        counter        = comm.allreduce(counter, MPI.SUM)
        total_icm_loss = comm.allreduce(total_icm_loss, MPI.SUM)
        self.status_dict["icm loss"] = total_icm_loss / counter


    def save(self):
        """
            Save all information required for a restart.
        """
        if self.test_mode:
            msg = "WARNING: save() was called while in test mode. Disregarding."
            rank_print(msg)
            return

        comm.barrier()

        self.actor.save(self.state_path)
        self.critic.save(self.state_path)

        if self.use_icm:
            self.icm_model.save(self.state_path)

        if self.save_env_info and self.env != None:
            self.env.save_info(self.state_path)

        if self.normalize_values:
            self.value_normalizer.save_info(self.state_path)

        file_name  = "state_{}.pickle".format(rank)
        state_file = os.path.join(self.state_path, file_name)
        with open(state_file, "wb") as out_f:
            pickle.dump(self.status_dict, out_f,
                protocol=pickle.HIGHEST_PROTOCOL)

        comm.barrier()

    def load(self):
        """
            Load all information required for a restart.
        """
        self.actor.load(self.state_path)
        self.critic.load(self.state_path)

        if self.use_icm:
            self.icm_model.load(self.state_path)

        if self.save_env_info and self.env != None:
            self.env.load_info(self.state_path)

        if self.normalize_values:
            self.value_normalizer.load_info(self.state_path)

        if self.test_mode:
            file_name  = "state_0.pickle"
        else:
            file_name  = "state_{}.pickle".format(rank)

        state_file = os.path.join(self.state_path, file_name)

        with open(state_file, "rb") as in_f:
            tmp_status_dict = pickle.load(in_f)

        return tmp_status_dict

    def set_test_mode(self, test_mode):
        """
            Enable or disable test mode in all required modules.

            Arguments:
                test_mode    A bool representing whether or not to enable
                             test_mode.
        """
        for module in self.test_mode_dependencies:
            module.test_mode = test_mode

    def __getstate__(self):
        """
            Override the getstate method for pickling. We want everything
            but the environment since we can't guarantee that the env can
            be pickled.

            Returns:
                The state dictionary minus the environment.
        """
        state = self.__dict__.copy()
        del state["env"]
        return state

    def __setstate__(self, state):
        """
            Override the setstate method for pickling. We want everything
            but the environment since we can't guarantee that the env can
            be pickled.

            Arguments:
                The state loaded from a pickled PPO object.
        """
        self.__dict__.update(state)
        self.env = None

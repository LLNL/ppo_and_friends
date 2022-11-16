import sys
import gc
import pickle
import numpy as np
import os
from copy import deepcopy
import torch
from torch import nn
from torch.utils.data import DataLoader
from ppo_and_friends.utils.misc import RunningStatNormalizer
from ppo_and_friends.utils.iteration_mappers import *
from ppo_and_friends.utils.misc import update_optimizer_lr
from ppo_and_friends.policies.utils import generate_policy
from ppo_and_friends.environments.wrapper_utils import wrap_environment
from ppo_and_friends.environments.filter_wrappers import RewardNormalizer, ObservationNormalizer
from ppo_and_friends.utils.mpi_utils import broadcast_model_parameters, mpi_avg_gradients
from ppo_and_friends.utils.mpi_utils import mpi_avg
from ppo_and_friends.utils.mpi_utils import rank_print, set_torch_threads
from ppo_and_friends.utils.misc import format_seconds
import time
from gym.spaces import Box, Discrete, MultiDiscrete, MultiBinary
from mpi4py import MPI

comm      = MPI.COMM_WORLD
rank      = comm.Get_rank()
num_procs = comm.Get_size()

# TODO:
#  1. let's allow each policy to have its own lr,
#     entropy_weight, gamma, icm_beta, reward weights, target_kl,
#     surr_clip, gradient_clip, bootstrap_clip, 
class PPO(object):

    def __init__(self,
                 env_generator,
                 device,
                 random_seed,
                 policy_settings,
                 policy_mapping_fn,
                 envs_per_proc       = 1,
                 lr                  = 3e-4,
                 min_lr              = 1e-4,
                 lr_dec              = None,
                 entropy_weight      = 0.01,
                 min_entropy_weight  = 0.01,
                 entropy_dec         = None,
                 max_ts_per_ep       = 200,
                 batch_size          = 256,
                 ts_per_rollout      = num_procs * 1024,
                 gamma               = 0.99,
                 epochs_per_iter     = 10,
                 surr_clip           = 0.2,
                 gradient_clip       = 0.5,
                 icm_beta            = 0.8,
                 ext_reward_weight   = 1.0,
                 intr_reward_weight  = 1.0,
                 target_kl           = 100.,
                 normalize_adv       = True,
                 normalize_obs       = True,
                 normalize_rewards   = True,
                 normalize_values    = True,
                 obs_clip            = None,
                 reward_clip         = None,
                 render              = False,
                 load_state          = False,
                 state_path          = "./",
                 save_every          = 1,
                 pickle_class        = False,
                 use_soft_resets     = False,
                 obs_augment         = False,
                 test_mode           = False):
        """
            Initialize the PPO trainer.

            Parameters:
                 env_generator        A function that creates instances of
                                      the environment to learn from.
                 device               A torch device to use for training.
                 random_seed          A random seed to use.
                 envs_per_proc        The number of environment instances each
                                      processor owns.
                 icm_network          The network to use for ICM applications.
                 lr                   The initial learning rate.
                 min_lr               The minimum learning rate.
                 lr_dec               A class that inherits from the
                                      IterationMapper class located in
                                      utils/iteration_mappers.py.
                                      This class has a decrement function that
                                      will be used to updated the learning rate.
                 entropy_dec          A class that inherits from the
                                      IterationMapper class located in
                                      utils/iteration_mappers.py.
                                      This class has a decrement function that
                                      will be used to updated the entropy
                                      weight.
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
                 save_every           Save every save_every iterations.
                 pickle_class         When enabled, the entire PPO class will
                                      be pickled and saved into the output
                                      directory after it's been initialized.
                 use_soft_resets      Use "soft resets" during rollouts.
                 obs_augment          This is a funny option that can only be
                                      enabled with environments that have a
                                      "observation_augment" method defined.
                                      When enabled, this method will be used to
                                      augment observations into batches of
                                      observations that all require the same
                                      treatment (a single action).
                 test_mode            Most of this class is not used for
                                      testing, but some of its attributes are.
                                      Setting this to True will enable test
                                      mode.
        """
        set_torch_threads()

        #
        # Divide the ts per rollout up among the processors. Make an adjustment
        # if needed.
        #
        orig_ts        = ts_per_rollout
        ts_per_rollout = int(ts_per_rollout / num_procs)
        if rank == 0 and (orig_ts % num_procs) > 0:
            msg  = "WARNING: {} timesteps per rollout ".format(ts_per_rollout)
            msg += "cannot be evenly distributed across "
            msg += "{} processors. The timesteps per ".format(num_procs)
            msg += "rollout have been adjusted for effecient distribution. "
            msg += "The new timesteps per rollout is "
            msg += "{}.".format(ts_per_rollout * num_procs)
            rank_print(msg)

        if not test_mode:
            rank_print("ts_per_rollout per rank: ~{}".format(ts_per_rollout))

        env, self.status_dict = wrap_environment(
            env_generator     = env_generator,
            envs_per_proc     = envs_per_proc,
            random_seed       = random_seed,
            obs_augment       = obs_augment,
            normalize_obs     = normalize_obs,
            normalize_rewards = normalize_rewards,
            obs_clip          = obs_clip,
            reward_clip       = reward_clip,
            gamma             = gamma,
            test_mode         = test_mode)

        #
        # If we're normalizing, we need to save the stats for reproduction
        # when in inference.
        #
        self.save_env_info = False
        if (env.has_wrapper(RewardNormalizer) or
            env.has_wrapper(ObservationNormalizer)):
            self.save_env_info = True

        #
        # FIXME: If we vectorize multi-agent environments, we'll need to
        # change this here. I think it will actually be simpler, because we
        # won't be treating agents as environments.
        #
        # Funny business: we're taking over our "environments per processor"
        # code when using multi-agent environments. Each agent is basically
        # thought of as an environment instance. As a result, our timesteps
        # will be divided by the number of agents during the rollout, which
        # is what we want when envs_per_proc > 1, but it's not what we
        # want when num_agents > 1.
        #
        #if #is_multi_agent:
        #   # ts_per_rollout *= env.get_num_agents()
        #   # self.num_agents = env.get_num_agents()

        #
        # When we toggle test mode on/off, we need to make sure to also
        # toggle this flag for any modules that depend on it.
        #
        self.test_mode_dependencies = [env]
        self.pickle_safe_test_mode_dependencies = []

        #
        # FIXME: we need to handle multiple agent classes with
        # different action spaces. I think we want the observation
        # spaces to remain consistent, though.
        #
        action_space = env.action_space

        if lr_dec == None:
            self.lr_dec = LinearDecrementer(
                max_iteration = 1,
                max_value     = lr,
                min_value     = min_lr)
        else:
            self.lr_dec = lr_dec

        if entropy_dec == None:
            self.entropy_dec = LinearDecrementer(
                max_iteration = 1,
                max_value     = entropy_weight,
                min_value     = min_entropy_weight)
        else:
            self.entropy_dec = entropy_dec

        #
        # Establish some class variables.
        #
        self.env                 = env
        self.device              = device
        self.state_path          = state_path
        self.render              = render
        self.icm_beta            = icm_beta
        self.ext_reward_weight   = ext_reward_weight
        self.intr_reward_weight  = intr_reward_weight
        self.min_lr              = min_lr
        self.max_ts_per_ep       = max_ts_per_ep
        self.batch_size          = batch_size
        self.ts_per_rollout      = ts_per_rollout
        self.gamma               = gamma
        self.target_kl           = target_kl
        self.epochs_per_iter     = epochs_per_iter
        self.surr_clip           = surr_clip
        self.gradient_clip       = gradient_clip
        self.entropy_weight      = entropy_weight
        self.min_entropy_weight  = min_entropy_weight
        self.prev_top_window     = -np.finfo(np.float32).max
        self.save_every          = save_every
        self.normalize_adv       = normalize_adv
        self.normalize_rewards   = normalize_rewards
        self.normalize_obs       = normalize_obs
        self.normalize_values    = normalize_values
        self.lr                  = lr
        self.use_soft_resets     = use_soft_resets
        self.obs_augment         = obs_augment
        self.test_mode           = test_mode
        self.actor_obs_shape     = self.env.observation_space.shape
        self.policy_mapping_fn   = policy_mapping_fn

        self.policies = {}
        for policy_id in policy_settings:
            settings = policy_settings[policy_id]

            self.policies[policy_id] = \
                generate_policy(
                    policy_name              = str(policy_id),
                    policy_class             = settings[0],
                    actor_observation_space  = settings[1],
                    critic_observation_space = settings[2],
                    action_space             = settings[3],
                    test_mode                = test_mode,
                    **settings[4])

        for agent_id in self.env.agent_ids:
            policy_id = self.policy_mapping_fn(agent_id)
            self.policies[policy_id].register_agent(agent_id)

        #
        # Create a dictionary to track the status of training.
        # These entries can be general, agent specific, or
        # policy specific.
        #
        max_int = np.iinfo(np.int32).max

        self.status_dict["general"] = {}
        self.status_dict["general"]["iteration"]      = 0
        self.status_dict["general"]["rollout time"]   = 0
        self.status_dict["general"]["train time"]     = 0
        self.status_dict["general"]["running time"]   = 0
        self.status_dict["general"]["timesteps"]      = 0
        self.status_dict["general"]["lr"]             = self.lr
        self.status_dict["general"]["entropy weight"] = self.entropy_weight
        self.status_dict["general"]["total episodes"] = 0
        self.status_dict["general"]["longest run"]    = 0

        for policy_id in self.policies:
            self.status_dict[policy_id] = {}
            self.status_dict[policy_id]["episode reward avg"]  = 0
            self.status_dict[policy_id]["extrinsic score avg"] = 0
            self.status_dict[policy_id]["top score"]           = -max_int
            self.status_dict[policy_id]["weighted entropy"]    = 0
            self.status_dict[policy_id]["actor loss"]          = 0
            self.status_dict[policy_id]["critic loss"]         = 0
            self.status_dict[policy_id]["kl avg"]              = 0
            self.status_dict[policy_id]["reward range"] = (max_int, -max_int)
            self.status_dict[policy_id]["obs range"]    = (max_int, -max_int)

        #
        # Value normalization is discussed in multiple papers, so I'm not
        # going to reference one in particular. In general, the idea is
        # to normalize the targets of the critic network using a running
        # average. The output of the critic then needs to be de-normalized
        # for calcultaing advantages. We track separate normalizers for
        # each policy.
        #
        if normalize_values:
            self.value_normalizers = {}

            for policy_id in policy_settings:
                self.value_normalizers[policy_id] = RunningStatNormalizer(
                    name      = "{}-value_normalizer".format(policy_id),
                    device    = self.device,
                    test_mode = test_mode)

            self.test_mode_dependencies.append(self.value_normalizers)
            self.pickle_safe_test_mode_dependencies.append(
                self.value_normalizers)

        for policy_id in self.policies:
            self.policies[policy_id].to(self.device)

        self.test_mode_dependencies.append(self.policies)
        self.pickle_safe_test_mode_dependencies.append(self.policies)

        for policy_id in self.policies:
            if self.policies[policy_id].enable_icm:
                self.status_dict[policy_id]["icm loss"] = 0
                self.status_dict[policy_id]["intrinsic score avg"] = 0

        if load_state:
            if not os.path.exists(state_path):
                msg  = "WARNING: state_path does not exist. Unable "
                msg += "to load state."
                rank_print(msg)
            else:
                rank_print("Loading state from {}".format(state_path))

                #
                # Let's ensure backwards compatibility with previous commits.
                #
                tmp_status_dict = self.load()

                for key in tmp_status_dict:
                    if key in self.status_dict:
                        self.status_dict[key] = tmp_status_dict[key]

                self.lr= min(self.status_dict["general"]["lr"], self.lr)
                self.status_dict["general"]["lr"] = self.lr

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

        #
        # Some methods (ICM) perform best if we can clone the environment,
        # but not all environments support this.
        #
        #FIXME: make sure we can still clone some environments.
        if test_mode:
            self.can_clone_env = False
        else:
            try:
                obs = self.env.reset()
                _, actions, _ = self.get_policy_actions(obs)
                cloned_env   = deepcopy(self.env)
                cloned_env.step(actions)
                self.can_clone_env = True
            except:
                self.can_clone_env = False

        rank_print("Can clone environment: {}".format(self.can_clone_env))

    def get_policy_actions(self, obs):
        """
        """
        raw_actions = {}
        actions     = {}
        log_probs   = {}

        for agent_id in obs:
            policy_id = self.policy_mapping_fn(agent_id)

            raw_action, action, log_prob = \
                self.policies[policy_id].get_action(obs[agent_id])

            raw_actions[agent_id] = raw_action
            actions[agent_id]     = action
            log_probs[agent_id]   = log_prob

        return raw_actions, actions, log_probs

    def get_policy_actions_from_aug_obs(self, obs):
        """
        """
        raw_actions = {} 
        actions     = {}
        log_probs   = {}

        for agent_id in obs:
            policy_id = self.policy_mapping_fn(agent_id)

            obs_slice = obs[agent_id][0:1]
            raw_action, action, log_prob = \
                self.policies[policy_id].get_action(obs_slice)

            raw_actions[agent_id] = raw_action
            actions[agent_id]     = action
            log_probs[agent_id]   = log_prob

        return raw_actions, actions, log_probs

    def get_policy_values(self, obs):
        """
        """
        values = {}

        for agent_id in obs:
            policy_id        = self.policy_mapping_fn(agent_id)
            value            = self.policies[policy_id].critic(obs[agent_id])
            values[agent_id] = value

        return values

    def get_natural_reward(self, info):
        """
        """
        have_nat_reward = False
        natural_reward  = {}
        first_agent     = next(iter(info))
        batch_size      = info[first_agent].size

        if "natural reward" in info[first_agent][0]:
            have_nat_reward = True
        else:
            return have_nat_reward, natural_reward

        if have_nat_reward:
            for agent_id in info:
                natural_reward[agent_id] = np.zeros((batch_size, 1))
                for b_idx in range(batch_size):
                    natural_reward[agent_id][b_idx] = \
                        info[agent_id][b_idx]["natural reward"]

        return have_nat_reward, natural_reward

    #FIXME: should we perform these operations in-place instead?
    def get_detached_dict(self, attached):
        """
        """
        detached = {}

        for agent_id in attached:
            detached[agent_id] = \
                attached[agent_id].detach().cpu().numpy()

        return detached

    def get_denormalized_values(self, values):
        """
        """
        denorm_values = {}

        for agent_id in values:
            policy_id = self.policy_mapping_fn(agent_id)
            value     = values[agent_id]
            value     = self.value_normalizers[policy_id].denormalize(value)
            denorm_values[agent_id] = value

        return denorm_values

    def get_normalized_values(self, values):
        """
        """
        norm_values = {}

        for agent_id in values:
            policy_id = self.policy_mapping_fn(agent_id)
            value     = values[agent_id]
            value     = self.value_normalizers[policy_id].normalize(value)
            norm_values[agent_id] = value

        return norm_values

    def establish_non_terminal_dones(self,
                                     info,
                                     dones):
        """
        """
        first_agent = next(iter(dones))
        batch_size  = dones[first_agent].size
        non_terminal_dones = np.zeros(batch_size).astype(bool)
        have_non_terminal_dones = False

        #
        # Because we always death mask, any environment that's done for
        # one agent is done for them all.
        #
        for b_idx in range(batch_size):
            if "non-terminal done" in info[first_agent][b_idx]:
                ntd = info[first_agent][b_idx]["non-terminal done"]
                non_terminal_dones[b_idx] = ntd
                have_non_terminal_dones   = ntd or have_non_terminal_dones

        if have_non_terminal_dones:
            for agent_id in dones:
                dones[agent_id][non_terminal_dones] = False

        return np.where(non_terminal_dones)[0]

    def np_dict_to_tensor_dict(self, obs):
        """
        """
        tensor_dict = {}

        for agent_id in obs:
            tensor_dict[agent_id] = torch.tensor(obs[agent_id],
                dtype=torch.float).to(self.device)

        return tensor_dict

    def apply_intrinsic_rewards(self,
                                rewards,
                                prev_obs,
                                obs,
                                actions):
        """
        """
        intr_rewards = {}

        for agent_id in obs:
            policy_id = self.policy_mapping_fn(agent_id)

            if self.policies[policy_id].enable_icm:
                intr_rewards[agent_id] = \
                    self.policies[policy_id].get_intrinsic_reward(
                        prev_obs[agent_id],
                        obs[agent_id],
                        actions[agent_id])

                rewards[agent_id] = rewards[agent_id] + intr_rewards[agent_id]
            else:
                intr_rewards[agent_id] = 0.0

        return rewards, intr_rewards

    def apply_reward_weight(self,
                            rewards,
                            weight):
        """
        """
        for agent_id in rewards:
            rewards[agent_id] *= weight

        return rewards

    def get_done_envs(self,
                      dones):
        """
        """
        first_id   = next(iter(dones))
        batch_size = dones[first_id].size
        done_envs  = np.zeros(batch_size).astype(bool)

        #
        # Because we always death mask, any agent that has a done environment
        # means that all agents are done in that same environment.
        #
        where_done     = np.where(dones[first_id])[0]
        where_not_done = np.where(~dones[first_id])[0]

        return where_done, where_not_done


    def print_status(self):
        """
            Print out statistics from our status_dict.
        """
        rank_print("\n--------------------------------------------------------")
        rank_print("Status Report:")
        for key in self.status_dict["general"]:

            if key in ["running time", "rollout time", "train time"]:
                pretty_time = format_seconds(self.status_dict["general"][key])
                rank_print("  {}: {}".format(key, pretty_time))
            else:
                rank_print("  {}: {}".format(key,
                    self.status_dict["general"][key]))

        for policy_id in self.policies:
            rank_print("  {} policy:".format(policy_id))
            for key in self.status_dict[policy_id]:
                rank_print("    {}: {}".format(key,
                    self.status_dict[policy_id][key]))

        rank_print("--------------------------------------------------------")

    def update_learning_rate(self,
                             iteration,
                             timestep):
        """
            Update the learning rate. This relies on the lr_dec function,
            which expects an iteration and returns an updated learning rate.

            Arguments:
                iteration    The current iteration of training.
        """

        self.lr = self.lr_dec(
            iteration = iteration,
            timestep  = timestep)

        for key in self.policies:
            self.policies[key].update_learning_rate(self.lr)

        # FIXME: do we really need this anymore? It's really just a way
        # of debugging. If our status dict changes to show info for each
        # policy, then we could still use this.
        #self.status_dict["lr"] = self.policy.actor_optim.param_groups[0]["lr"]
        self.status_dict["general"]["lr"] = self.lr

    def update_entropy_weight(self,
                              iteration,
                              timestep):
        """
            Update the entropy weight. This relies on the entropy_dec function,
            which expects an iteration and returns an updated entropy weight.

            Arguments:
                iteration    The current iteration of training.
        """
        self.entropy_weight = self.entropy_dec(
            iteration = iteration,
            timestep  = timestep)

        self.status_dict["general"]["entropy weight"] = self.entropy_weight

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
        start_time = time.time()

        if self.env ==  None:
            msg  = "ERROR: unable to perform rollout due to the environment "
            msg += "being of type None. This is likey due to loading the "
            msg += "PPO class from a pickled state."
            rank_print(msg)
            comm.Abort()

        for key in self.policies:
            self.policies[key].initialize_dataset()

        total_episodes   = 0.0
        total_rollout_ts = 0
        longest_run      = 0

        for key in self.policies:
            self.policies[key].eval()

        #
        # TODO: soft resets might cause rollouts to start off in "traps"
        # that are impossible to escape. We might be able to handle this
        # more intelligently.
        #
        if self.use_soft_resets:
            initial_reset_func = self.env.soft_reset
        else:
            initial_reset_func = self.env.reset

        #
        # FIXME: In the single agent case, env_batch_size is the number
        # environments per processor (vectorized environment). In the
        # multi-agent case, env_batch_size is the number of agents in
        # the environment.
        # In the refactored MA setting, agents don't need to all step
        # at once. We'll also have a dictionary mapping agents to their
        # observations. Since there might be situations where only a
        # single agent acts, we might want to revisit vectorizing the
        # multi-agent environments.
        #
        # ALSO, our env_batch_size might not be adhered to when agents
        # aren't stepping in sync... How do we handle that case?
        # Maybe we keep track of the max number of agents that could
        # step at one time...
        #
        obs, global_obs    = initial_reset_func()
        env_batch_size     = self.env.get_batch_size()

        top_rollout_score  = {}
        rollout_max_reward = {}
        rollout_min_reward = {}
        rollout_max_obs    = {}
        rollout_min_obs    = {}
        ep_nat_rewards     = {}
        ep_rewards         = {}
        ep_intr_rewards    = {}
        total_ext_rewards  = {}
        total_intr_rewards = {}
        total_rewards      = {}
        agents_per_policy  = {}

        for policy_id in self.policies:
            top_rollout_score[policy_id]  = -np.finfo(np.float32).max
            rollout_max_reward[policy_id] = -np.finfo(np.float32).max
            rollout_min_reward[policy_id] = np.finfo(np.float32).max
            rollout_max_obs[policy_id]    = -np.finfo(np.float32).max
            rollout_min_obs[policy_id]    = np.finfo(np.float32).max
            ep_nat_rewards[policy_id]     = np.zeros((env_batch_size, 1))
            ep_intr_rewards[policy_id]    = np.zeros((env_batch_size, 1))
            ep_rewards[policy_id]         = np.zeros((env_batch_size, 1))
            total_ext_rewards[policy_id]  = np.zeros((env_batch_size, 1))
            total_intr_rewards[policy_id] = np.zeros((env_batch_size, 1))
            total_rewards[policy_id]      = np.zeros((env_batch_size, 1))
            agents_per_policy[policy_id]  = 0

        episode_lengths = np.zeros(env_batch_size).astype(np.int32)
        ep_ts           = np.zeros(env_batch_size).astype(np.int32)

        for key in self.policies:
            self.policies[key].initialize_episodes(
                env_batch_size, self.status_dict)

        #
        # TODO: If we're using multiple environments, we can end up going over
        # our requested limits here... We could get around this by truncating
        # the batch when necessary.
        # UPDATE: is this still true? We've made some changes that may have
        # had an effect on this.
        #
        while total_rollout_ts < self.ts_per_rollout:

            for policy_id in agents_per_policy:
                agents_per_policy[policy_id] = 0

            ep_ts += 1

            if self.render:
                self.env.render()

            total_rollout_ts += env_batch_size
            episode_lengths  += 1

            start = time.time()#FIXME

            if self.obs_augment:
                raw_action, action, log_prob = \
                    self.get_policy_actions_from_aug_obs(obs)
            else:
                raw_action, action, log_prob = \
                    self.get_policy_actions(obs)

            stop = time.time()#FIXME
            print(f"time: {stop - start}")

            critic_obs = self.np_dict_to_tensor_dict(global_obs)

            value = self.get_policy_values(critic_obs)

            if self.normalize_values:
                value = self.get_denormalized_values(value)

            #
            # Note that we have global observations as well as local.
            # When we're learning from a multi-agent environment, we
            # feed a "global state" to the critic. This is called
            # Centralized Training Decentralized Execution (CTDE).
            # arXiv:2006.07869v4
            #
            prev_obs        = deepcopy(obs)
            prev_global_obs = deepcopy(global_obs)

            #
            # The returned objects are dictionaries mapping agent ids
            # to np arrays. Each element of the numpy array represents
            # the results from a single environment.
            #
            obs, global_obs, ext_reward, done, info = self.env.step(action)

            #
            # Non-terminal dones are interesting cases. We need to
            # first find them and then set any corresponding "dones"
            # in the done array to false. This is because we treat
            # these non-terminal done states as needing to end without
            # entering a terminal state.
            #
            where_non_terminal = \
                self.establish_non_terminal_dones(
                    info,
                    done)

            have_non_terminal_dones = where_non_terminal.size > 0

            #
            # In the observational augment case, our action is a single action,
            # but our return values are all batches. We need to tile the
            # actions into batches as well.
            #
            if self.obs_augment:
                #FIXME: refactor this
                batch_size   = obs.shape[0]

                action_shape = (batch_size,) + action.shape[1:]
                action       = np.tile(action.flatten(), batch_size)
                action       = action.reshape(action_shape)

                raw_action   = np.tile(raw_action.flatten(), batch_size)
                raw_action   = raw_action.reshape(action_shape)

                lp_shape     = (batch_size,) + log_prob.shape[1:]
                log_prob     = np.tile(log_prob.flatten(), batch_size)
                log_prob     = log_prob.reshape(lp_shape)

            value = self.get_detached_dict(value)

            #
            # If any of our wrappers are altering the rewards, there should
            # be an unaltered version in the info.
            #
            have_nat_reward, natural_reward = self.get_natural_reward(info)

            if not have_nat_reward:
                natural_reward = deepcopy(ext_reward)

            self.apply_reward_weight(ext_reward, self.ext_reward_weight)

            #
            # If we're using the ICM, we need to do some extra work here.
            # This amounts to adding "curiosity", aka intrinsic reward,
            # to out extrinsic reward.
            #
            reward, intr_rewards = self.apply_intrinsic_rewards(
                ext_reward,
                prev_obs,
                obs,
                action)

            ep_obs = deepcopy(obs)

            where_done, where_not_done = self.get_done_envs(done)
            done_count = where_done.size

            for agent_id in action:
                if done_count > 0:
                    for done_idx in where_done:
                        ep_obs[agent_id][done_idx] = \
                            info[agent_id][done_idx]["terminal observation"]

                policy_id = self.policy_mapping_fn(agent_id)

                self.policies[policy_id].add_episode_info(
                    agent_id             = agent_id,
                    global_observations  = prev_global_obs[agent_id],
                    observations         = prev_obs[agent_id],
                    next_observations    = ep_obs[agent_id],
                    raw_actions          = raw_action[agent_id],
                    actions              = action[agent_id],
                    values               = value[agent_id],
                    log_probs            = log_prob[agent_id],
                    rewards              = reward[agent_id],
                    where_done           = where_done)

                rollout_max_reward[policy_id] = \
                    max(rollout_max_reward[policy_id], reward[agent_id].max())

                rollout_min_reward[policy_id] = \
                    min(rollout_min_reward[policy_id], reward[agent_id].min())

                rollout_max_obs[policy_id]    = \
                    max(rollout_max_obs[policy_id], obs[agent_id].max())

                rollout_min_obs[policy_id]    = \
                    min(rollout_min_obs[policy_id], obs[agent_id].min())

                ep_rewards[policy_id]        += reward[agent_id]
                ep_nat_rewards[policy_id]    += natural_reward[agent_id]
                ep_intr_rewards[policy_id]   += intr_rewards[agent_id]
                agents_per_policy[policy_id] += 1

            #
            # Since each policy can have multiple agents, we average
            # the scores to get a more interpretable value.
            #
            for policy_id in agents_per_policy:
                ep_rewards[policy_id]      /= agents_per_policy[policy_id]
                ep_nat_rewards[policy_id]  /= agents_per_policy[policy_id]
                ep_intr_rewards[policy_id] /= agents_per_policy[policy_id]

            #
            # Episode end cases.
            #  1. An episode has reached a "done" state.
            #  2. An episode has reached the maximum allowable timesteps.
            #  3. An episode has reached a non-terminal done state.
            #
            # Case 1.
            # We handle any episodes that have reached a terminal done state.
            # In these cases, the environment cannot proceed any further.
            #
            if done_count > 0:
                #
                # Every agent has at least one done environment.
                #
                # FIXME: agents now share episodes, so no single agent can end
                # an episode.
                # NOTE: if one agent is done, they're all done (death masking).
                for agent_id in done:
                    policy_id = self.policy_mapping_fn(agent_id)

                    self.policies[policy_id].end_episodes(
                        agent_id        = agent_id,
                        env_idxs        = where_done,
                        episode_lengths = episode_lengths,
                        terminal        = np.ones(done_count).astype(bool),
                        ending_values   = np.zeros(done_count),
                        ending_rewards  = np.zeros(done_count),
                        status_dict     = self.status_dict)

                    top_rollout_score[policy_id] = \
                        max(top_rollout_score[policy_id],
                        ep_nat_rewards[policy_id][where_done].max())

                    total_ext_rewards[policy_id][where_done] += \
                        ep_nat_rewards[policy_id][where_done]

                    if self.policies[policy_id].enable_icm:
                        total_intr_rewards[policy_id][where_done] += \
                            ep_intr_rewards[policy_id][where_done]

                    total_rewards[policy_id] += \
                        ep_rewards[policy_id][where_done]

                    ep_rewards[policy_id][where_done]      = 0
                    ep_nat_rewards[policy_id][where_done]  = 0
                    ep_intr_rewards[policy_id][where_done] = 0


                longest_run = max(longest_run,
                    episode_lengths[where_done].max())

                episode_lengths[where_done]    = 0
                total_episodes                += done_count
                ep_ts[where_done]              = 0

            #
            # Cases 2 and 3.
            # We handle episodes that have reached or exceeded the maximum
            # number of timesteps allowed, but they haven't yet reached a
            # terminal done state. This is also very similar to reaching
            # an environment triggered non-terminal done state, so we handle
            # them at the same time (identically).
            # Since the environment can continue, we can take this into
            # consideration when calculating the reward.
            #
            ep_max_reached = ((ep_ts == self.max_ts_per_ep).any() and
                where_not_done.size > 0)

            if (ep_max_reached or
                total_rollout_ts >= self.ts_per_rollout or
                have_non_terminal_dones):

                if total_rollout_ts >= self.ts_per_rollout:
                    where_maxed = np.arange(env_batch_size)
                else:
                    where_maxed = np.where(ep_ts >= self.max_ts_per_ep)[0]

                where_maxed = np.setdiff1d(where_maxed, where_done)
                where_maxed = np.concatenate((where_maxed, where_non_terminal))
                where_maxed = np.unique(where_maxed)

                critic_obs  = self.np_dict_to_tensor_dict(global_obs)
                next_value  = self.get_policy_values(critic_obs)

                if self.normalize_values:
                    next_value = self.get_denormalized_values(next_value)

                next_reward = self.get_detached_dict(next_value)

                #
                # Tricky business:
                # Typically, we just use the result of our critic to
                # bootstrap the expected reward. This is problematic
                # with ICM because we can't really expect our critic to
                # learn about "surprise". I don't know of any perfect
                # ways to handle this, but here are some ideas:
                #
                #     1. Just use the value anyways. As long as the
                #        max ts per episode is long enough, we'll
                #        hopefully see enough intrinsic reward to
                #        learn a good policy. In my experience, this
                #        works, but the learned policies can be a bit
                #        unstable.
                #     2. If we can clone the environment, we can take
                #        an extra step with the clone to get the
                #        intrinsic reward, and we can decide what to
                #        do with this. Approaches that integrate this
                #        method tend to learn more stable policies.
                #
                # If we have this intrinsic reward from a clone step,
                # we can hand wavily calcluate a "surprise" by taking
                # the difference between the average intrinsic reward
                # and the one we get. Adding that to the critic's
                # output can act as an extra surprise bonus.
                #
                maxed_count = where_maxed.size

                for agent_id in next_reward:
                    policy_id = self.policy_mapping_fn(agent_id)

                    if self.policies[policy_id].enable_icm:
                        if self.can_clone_env:
                            intr_rewards[agent_id] = \
                                self.policies[policy_id].get_cloned_intrinsic_reward(
                                    obs            = obs[agent_id],
                                    obs_augment    = self.obs_augment)

                        ism = self.status_dict[policy_id]["intrinsic score avg"]
                        surprise = intr_rewards[agent_id][where_maxed] - ism
                        next_reward[agent_id] += surprise
                    
                    self.policies[policy_id].end_episodes(
                        agent_id        = agent_id,
                        env_idxs        = where_maxed,
                        episode_lengths = episode_lengths,
                        terminal        = np.zeros(maxed_count).astype(bool),
                        ending_values   = next_value[agent_id],
                        ending_rewards  = next_reward[agent_id],
                        status_dict     = self.status_dict)

                if total_rollout_ts >= self.ts_per_rollout:

                    #
                    # ts_before_ep are the timesteps before the current
                    # episode. We use this to calculate the average episode
                    # length (before the current one). If we didn't finish
                    # this episode, we can then calculate a rough estimate
                    # of how far we were in the episode as a % of the avg.
                    #
                    combined_ep_len = episode_lengths.sum()
                    ts_before_ep    = self.ts_per_rollout - combined_ep_len
                    ts_before_ep    = max(ts_before_ep, 0)
                    current_total   = total_episodes

                    if current_total == 0:
                        current_total = 1.0

                    if ts_before_ep == 0:
                        avg_ep_len = combined_ep_len / env_batch_size
                    else:
                        avg_ep_len = ts_before_ep / current_total

                    ep_perc         = episode_lengths / avg_ep_len
                    total_episodes += ep_perc.sum()


                    for policy_id in self.policies:
                        total_ext_rewards[policy_id] += \
                            ep_nat_rewards[policy_id]

                        total_rewards[policy_id] += \
                            ep_rewards[policy_id]

                        if self.policies[policy_id].enable_icm:
                            total_intr_rewards[policy_id] += \
                                ep_intr_rewards[policy_id]

                        top_rollout_score[policy_id] = \
                            max(top_rollout_score[policy_id],
                            ep_nat_rewards[policy_id].max())

                ep_ts[where_maxed] = 0

            longest_run = max(longest_run,
                episode_lengths.max())

        #
        # Update the status dictionary.
        #
        total_episodes = comm.allreduce(total_episodes, MPI.SUM)

        for policy_id in self.policies:
            #
            # We didn't complete any episodes, so let's just take the top score
            # from our incomplete episode's scores.
            #
            if total_episodes <= 1.0:
                top_rollout_score[policy_id] = max(top_rollout_score[policy_id],
                    ep_nat_rewards[policy_id].max())

            top_score = max(top_rollout_score[policy_id],
                self.status_dict[policy_id]["top score"])

            top_score = comm.allreduce(top_score, MPI.MAX)

            #
            # If we're normalizing, we don't really want to keep track
            # of the largest and smallest ever seen, because our range will
            # fluctuate with normalization. When we aren't normalizing, the
            # the global range is accurate and useful.
            #
            if not self.normalize_rewards:
                max_reward = max(self.status_dict[policy_id]["reward range"][1],
                    rollout_max_reward[policy_id])

                min_reward = min(self.status_dict[policy_id]["reward range"][0],
                    rollout_min_reward[policy_id])
            else:
                max_reward = rollout_max_reward[policy_id]
                min_reward = rollout_min_reward[policy_id]

            max_reward = comm.allreduce(max_reward, MPI.MAX)
            min_reward = comm.allreduce(min_reward, MPI.MIN)

            if not self.normalize_obs:
                max_obs = max(self.status_dict[policy_id]["obs range"][1],
                    rollout_max_obs[policy_id])

                min_obs = min(self.status_dict[policy_id]["obs range"][0],
                    rollout_min_obs[policy_id])
            else:
                max_obs = rollout_max_obs[policy_id]
                min_obs = rollout_min_obs[policy_id]

            max_obs = comm.allreduce(max_obs, MPI.MAX)
            min_obs = comm.allreduce(min_obs, MPI.MIN)

            ext_reward_sum = total_ext_rewards[policy_id].sum()
            ext_reward_sum = comm.allreduce(ext_reward_sum, MPI.SUM)

            agent_rewards  = total_rewards[policy_id].sum()
            agent_rewards  = comm.allreduce(agent_rewards, MPI.SUM)

            running_ext_score = ext_reward_sum / total_episodes
            running_score     = agent_rewards / total_episodes
            rw_range          = (min_reward, max_reward)
            obs_range         = (min_obs, max_obs)

            self.status_dict[policy_id]["episode reward avg"]  = running_score
            self.status_dict[policy_id]["extrinsic score avg"] = running_ext_score
            self.status_dict[policy_id]["top score"]           = top_score
            self.status_dict[policy_id]["obs range"]       = obs_range
            self.status_dict[policy_id]["reward range"]    = rw_range

            if self.policies[policy_id].enable_icm:
                intr_rewards = total_intr_rewards[policy_id].sum()
                intr_rewards = comm.allreduce(intr_rewards, MPI.SUM)

                ism = intr_rewards / (total_episodes/ env_batch_size)
                self.status_dict[policy_id]["intrinsic score avg"] = ism.item()

        longest_run      = comm.allreduce(longest_run, MPI.MAX)
        total_rollout_ts = comm.allreduce(total_rollout_ts, MPI.SUM)

        self.status_dict["general"]["total episodes"] += total_episodes
        self.status_dict["general"]["longest run"]     = longest_run
        self.status_dict["general"]["timesteps"]      += total_rollout_ts

        #
        # Finalize our datasets.
        #
        for key in self.policies:
            self.policies[key].finalize_dataset()

        comm.barrier()
        stop_time = time.time()
        self.status_dict["general"]["rollout time"] = stop_time - start_time

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
        ts_max     = self.status_dict["general"]["timesteps"] + num_timesteps
        iteration  = self.status_dict["general"]["iteration"]

        while self.status_dict["general"]["timesteps"] < ts_max:
            iter_start_time = time.time()

            self.rollout()

            self.update_learning_rate(
                self.status_dict["general"]["iteration"],
                self.status_dict["general"]["timesteps"])

            self.update_entropy_weight(
                self.status_dict["general"]["iteration"],
                self.status_dict["general"]["timesteps"])

            data_loaders = {}
            for key in self.policies:
                data_loaders[key] = DataLoader(
                    self.policies[key].dataset,
                    batch_size = self.batch_size,
                    shuffle    = True)

            train_start_time = time.time()

            for key in self.policies:
                self.policies[key].train()

            #
            # We train each policy separately.
            #
            for policy_id in data_loaders:

                for epoch_idx in range(self.epochs_per_iter):

                    #
                    # arXiv:2006.05990v1 suggests that re-computing the
                    # advantages before each new epoch helps mitigate issues
                    # that can arrise from "stale" advantages.
                    #
                    if epoch_idx > 0:
                        data_loaders[policy_id].dataset.recalculate_advantages()

                    self._ppo_batch_train(data_loaders[policy_id], policy_id)

                    #
                    # Early ending using KL. Why multiply by 1.5, you ask? I have
                    # no idea, really. It's a magic number that the folks at
                    # OpenAI are using.
                    #
                    comm.barrier()
                    if (self.status_dict[policy_id]["kl avg"] >
                        (1.5 * self.target_kl)):

                        msg  = "\nTarget KL of {} ".format(1.5 * self.target_kl)
                        msg += "has been reached. "
                        msg += "Ending early (after "
                        msg += "{} epochs)".format(epoch_idx + 1)
                        rank_print(msg)
                        break

                    if self.policies[policy_id].enable_icm:
                        self._icm_batch_train(data_loaders[policy_id], policy_id)

            #
            # We don't want to hange on to this memory as we loop back around.
            #
            for policy_id in self.policies:
                self.policies[policy_id].clear_dataset()

            del data_loaders
            gc.collect()

            now_time      = time.time()
            training_time = (now_time - train_start_time)
            self.status_dict["general"]["train time"] = now_time - train_start_time

            running_time = (now_time - iter_start_time)
            self.status_dict["general"]["running time"] += running_time

            iteration += 1
            self.status_dict["general"]["iteration"] = iteration

            self.print_status()

            if self.save_every > 0 and iteration % self.save_every == 0:
                self.save()

            comm.barrier()
            if self.lr <= 0.0:
                rank_print("Learning rate has bottomed out. Terminating early")
                break

        stop_time   = time.time()
        seconds     = (stop_time - start_time)
        pretty_time = format_seconds(seconds)
        rank_print("Time spent training: {}".format(pretty_time))

    def _ppo_batch_train(self, data_loader, policy_id):
        """
            Train our PPO networks using mini batches.

            Arguments:
                data_loader    A PyTorch data loader for a specific policy.
                policy_id      The id for the policy that we're training.
        """
        total_actor_loss  = 0
        total_critic_loss = 0
        total_entropy     = 0
        total_w_entropy   = 0
        total_kl          = 0
        counter           = 0

        for batch_data in data_loader:
            critic_obs, obs, _, raw_actions, _, advantages, log_probs, \
                rewards_tg, actor_hidden, critic_hidden, \
                actor_cell, critic_cell, batch_idxs = batch_data

            torch.cuda.empty_cache()

            if self.normalize_values:
                rewards_tg = \
                    self.value_normalizers[policy_id].normalize(rewards_tg)

            if obs.shape[0] == 1:
                continue

            #
            # In the case of lstm networks, we need to initialze our hidden
            # states to those that developed during the rollout.
            #
            if self.policies[policy_id].using_lstm:
                actor_hidden  = torch.transpose(actor_hidden, 0, 1)
                actor_cell    = torch.transpose(actor_cell, 0, 1)
                critic_hidden = torch.transpose(critic_hidden, 0, 1)
                critic_cell   = torch.transpose(critic_cell, 0, 1)

                self.policies[policy_id].actor.hidden_state  = (actor_hidden, actor_cell)
                self.policies[policy_id].critic.hidden_state = (critic_hidden, critic_cell)

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

            values, curr_log_probs, entropy = self.policies[policy_id].evaluate(
                critic_obs,
                obs,
                raw_actions)

            data_loader.dataset.values[batch_idxs] = values

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

                #FIXME
                std = nn.functional.softplus(self.policies[policy_id].actor.distribution.log_std)
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

            critic_loss        = nn.MSELoss()(values, rewards_tg)
            total_critic_loss += critic_loss.item()

            #
            # arXiv:2005.12729v1 suggests that gradient clipping can
            # have a positive effect on training.
            #
            #FIXME: cleanup
            self.policies[policy_id].actor_optim.zero_grad()
            actor_loss.backward(retain_graph = self.policies[policy_id].using_lstm)
            mpi_avg_gradients(self.policies[policy_id].actor)
            nn.utils.clip_grad_norm_(self.policies[policy_id].actor.parameters(),
                self.gradient_clip)
            self.policies[policy_id].actor_optim.step()

            self.policies[policy_id].critic_optim.zero_grad()
            critic_loss.backward(retain_graph = self.policies[policy_id].using_lstm)
            mpi_avg_gradients(self.policies[policy_id].critic)
            nn.utils.clip_grad_norm_(self.policies[policy_id].critic.parameters(),
                self.gradient_clip)
            self.policies[policy_id].critic_optim.step()

            #
            # The idea here is similar to re-computing advantages, but now
            # we want to update the hidden states before the next epoch.
            #
            if self.policies[policy_id].using_lstm:
                actor_hidden  = self.policies[policy_id].actor.hidden_state[0].detach().clone()
                critic_hidden = self.policies[policy_id].critic.hidden_state[0].detach().clone()

                actor_cell    = self.policies[policy_id].actor.hidden_state[1].detach().clone()
                critic_cell   = self.policies[policy_id].critic.hidden_state[1].detach().clone()

                actor_hidden  = torch.transpose(actor_hidden, 0, 1)
                actor_cell    = torch.transpose(actor_cell, 0, 1)
                critic_hidden = torch.transpose(critic_hidden, 0, 1)
                critic_cell   = torch.transpose(critic_cell, 0, 1)

                data_laoder.dataset.actor_hidden[batch_idxs]  = actor_hidden
                data_laoder.dataset.critic_hidden[batch_idxs] = critic_hidden

                data_laoder.dataset.actor_cell[batch_idxs]  = actor_cell
                data_laoder.dataset.critic_cell[batch_idxs] = critic_cell

            comm.barrier()
            counter += 1

        counter           = comm.allreduce(counter, MPI.SUM)
        total_entropy     = comm.allreduce(total_entropy, MPI.SUM)
        total_actor_loss  = comm.allreduce(total_actor_loss, MPI.SUM)
        total_critic_loss = comm.allreduce(total_critic_loss, MPI.SUM)
        total_kl          = comm.allreduce(total_kl, MPI.SUM)
        w_entropy         = total_entropy * self.entropy_weight

        self.status_dict[policy_id]["weighted entropy"] = w_entropy / counter
        self.status_dict[policy_id]["actor loss"] = \
            total_actor_loss / counter

        self.status_dict[policy_id]["critic loss"] = \
            total_critic_loss / counter

        self.status_dict[policy_id]["kl avg"] = total_kl / counter

    def _icm_batch_train(self, data_loader, policy_id):
        """
            Train our ICM networks using mini batches.

            Arguments:
                data_loader    A PyTorch data loader for a specific policy.
                policy_id      The id for the policy that we're training.
        """
        total_icm_loss = 0
        counter = 0

        for batch_data in data_loader:

            _, obs, next_obs, _, actions, _, _, _, _, _, _, _, _ =\
                batch_data

            torch.cuda.empty_cache()

            actions = actions.unsqueeze(1)

            _, inv_loss, f_loss = self.policies[policy_id].icm_model(obs, next_obs, actions)

            icm_loss = (((1.0 - self.icm_beta) * f_loss) +
                (self.icm_beta * inv_loss))

            total_icm_loss += icm_loss.item()

            self.policies[policy_id].icm_optim.zero_grad()
            icm_loss.backward()
            mpi_avg_gradients(self.policies[policy_id].icm_model)
            self.policies[policy_id].icm_optim.step()

            counter += 1
            comm.barrier()

        counter        = comm.allreduce(counter, MPI.SUM)
        total_icm_loss = comm.allreduce(total_icm_loss, MPI.SUM)
        self.status_dict[policy_id]["icm loss"] = total_icm_loss / counter

    def save(self):
        """
            Save all information required for a restart.
        """
        if self.test_mode:
            msg = "WARNING: save() was called while in test mode. Disregarding."
            rank_print(msg)
            return

        comm.barrier()

        for policy_id in self.policies:
            self.policies[policy_id].save(self.state_path)

        if self.save_env_info and self.env != None:
            self.env.save_info(self.state_path)

        if self.normalize_values:
            for policy_id in self.value_normalizers:
                self.value_normalizers[policy_id].save_info(self.state_path)

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

        for policy_id in self.policies:
            self.policies[policy_id].load(self.state_path)

        if self.save_env_info and self.env != None:
            self.env.load_info(self.state_path)

        if self.normalize_values:
            for policy_id in self.value_normalizers:
                self.value_normalizers[policy_id].load_info(self.state_path)

        if self.test_mode:
            file_name  = "state_0.pickle"
        else:
            file_name  = "state_{}.pickle".format(rank)

        state_file = os.path.join(self.state_path, file_name)

        #
        # There are cases where we initially train using X ranks, and we
        # later want to continue training using (X+k) ranks. In these cases,
        # let's copy rank 0's info to all ranks > X.
        #
        if not os.path.exists(state_file):
            file_name  = "state_0.pickle"
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
        self.test_mode = test_mode

        for module in self.test_mode_dependencies:
            module.test_mode = test_mode

    def __getstate__(self):
        """
            Override the getstate method for pickling. We only want to keep
            things that won't upset pickle. The environment is something
            that we can't guarantee can be pickled.

            Returns:
                The state dictionary minus the environment.
        """
        state = self.__dict__.copy()
        del state["env"]
        del state["test_mode_dependencies"]
        return state

    def __setstate__(self, state):
        """
            Override the setstate method for pickling.

            Arguments:
                The state loaded from a pickled PPO object.
        """
        self.__dict__.update(state)
        self.env = None
        self.test_mode_dependencies = self.pickle_safe_test_mode_dependencies

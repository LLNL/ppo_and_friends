import sys
import gc
import dill as pickle
import numpy as np
import os
from copy import deepcopy
import torch
from torch import nn
from torch.utils.data import DataLoader
from ppo_and_friends.utils.misc import RunningStatNormalizer
from ppo_and_friends.utils.misc import update_optimizer_lr
from ppo_and_friends.policies.utils import generate_policy
from ppo_and_friends.policies.mat_policy import MATPolicy
from ppo_and_friends.environments.wrapper_utils import wrap_environment
from ppo_and_friends.environments.filter_wrappers import RewardNormalizer, ObservationNormalizer
from ppo_and_friends.utils.mpi_utils import broadcast_model_parameters, mpi_avg_gradients
from ppo_and_friends.utils.mpi_utils import mpi_avg
from ppo_and_friends.utils.mpi_utils import rank_print, set_torch_threads
from ppo_and_friends.utils.misc import format_seconds
from ppo_and_friends.utils.schedulers import LinearStepScheduler, CallableValue, ChangeInStateScheduler, FreezeCyclingScheduler
from pathlib import Path
import time
from collections import OrderedDict
from mpi4py import MPI

comm      = MPI.COMM_WORLD
rank      = comm.Get_rank()
num_procs = comm.Get_size()

class EpisodeScores(object):

    def __init__(self,
                 env_batch_size,
                 policy_ids):
        """
        This class tracks episode scores across rollouts.
        NOTE: if an environment does not have a timestep termination,
        and the policies learn to avoid other types of terminations,
        you can end up with never ending episodes. This is expected.

        Parameters:
        -----------
        env_batch_size: int
            The batch size for environments on this rank.
        policy_ids: array-like
            The policy IDs in this game.
        """

        self.policy_ids      = policy_ids
        self.env_batch_size  = env_batch_size

        self.running_scores  = {}
        self.episode_count   = {}
        self.finished_scores = {}

        for policy_id in self.policy_ids:
            self.running_scores[policy_id]  = np.zeros((self.env_batch_size, 1))
            self.episode_count[policy_id]   = 0
            self.finished_scores[policy_id] = 0.0

    def add_scores(self, policy_id, scores):
        """
        Add scores for the given policy.

        Parameters:
        -----------
        policy_id: dict key
            The policy ID to add scores to.
        scores: np.ndarray 
            An array of scores having shape (envs_batch_size, 1).
        """
        self.running_scores[policy_id] += scores

    def end_episodes(self, policy_id, episode_idxs):
        """
        End episodes for a given policy.

        Parameters:
        -----------
        policy_id: dict key
            The policy to end episodes for.
        episode_idxs: np.ndarray
            An array of episode indices that have ended.
        """
        self.finished_scores[policy_id] += self.running_scores[policy_id][episode_idxs].sum()
        self.episode_count[policy_id]   += len(episode_idxs)
        self.running_scores[policy_id][episode_idxs] = 0.0

    def _clear_episodes(self, policy_id):
        """
        Clear finished episodes for a given policy.

        Parameters:
        -----------
        policy_id: dict key
            The policy to clear episodes for.
        """
        self.episode_count[policy_id]   = 0
        self.finished_scores[policy_id] = 0

    def get_mean_scores(self):
        """
        Get the mean scores of all finished episodes, and then clear those
        episodes from memory.

        Returns:
        --------
        A dict mapping policy IDs to mean episode scores for those policies.
        """
        scores = {}
        for policy_id in self.policy_ids:
            score_sums     = self.finished_scores[policy_id]
            total_episodes = comm.allreduce(self.episode_count[policy_id], MPI.SUM)
            total_scores   = comm.allreduce(score_sums, MPI.SUM)

            self._clear_episodes(policy_id)

            if total_episodes > 0:
                scores[policy_id] = total_scores / total_episodes

        return scores


class PPO(object):

    def __init__(self,
                 env_generator,
                 policy_settings,
                 policy_mapping_fn,
                 device              = 'cpu',
                 random_seed         = None,
                 envs_per_proc       = 1,
                 max_ts_per_ep       = 200,
                 batch_size          = 256,
                 ts_per_rollout      = 1024,
                 gamma               = 0.99,
                 epochs_per_iter     = 10,
                 ext_reward_weight   = 1.0,
                 normalize_adv       = True,
                 normalize_obs       = True,
                 normalize_rewards   = True,
                 normalize_values    = True,
                 obs_clip            = None,
                 reward_clip         = None,
                 recalc_advantages   = False,
                 render              = False,
                 frame_pause         = 0.0,
                 load_state          = False,
                 state_path          = "./saved_state",
                 pretrained_policies = {},
                 env_state           = None,
                 freeze_policies     = [],
                 freeze_scheduler    = None,
                 checkpoint_every    = 100,
                 save_train_scores   = True,
                 save_ep_scores      = True,
                 save_avg_ep_len     = True,
                 save_running_time   = True,
                 save_bs_info        = True,
                 pickle_class        = False,
                 soft_resets         = False,
                 obs_augment         = False,
                 test_mode           = False,
                 force_gc            = False,
                 policy_tag          = "latest",
                 verbose             = False,
                 **kw_args):
        """
        Initialize the PPO trainer.

        Parameters:
        -----------
        env_generator: function
            A function that creates instances of
            the environment to learn from.
        policy_settings: dict
            A dictionary containing RLLib-like
            policy settings.
        policy_mapping_fn: function
            A function mapping agent ids to
            policy ids.
        device: str
            A string representing the torch device to use for
            training and inference.
        random_seed: int
            A random seed to use.
        envs_per_proc: int
            The number of environment instances each
            processor owns.
        lr: float
            The initial learning rate.
        max_ts_per_ep: int
            The maximum timesteps to allow per episode.
        batch_size: int
            The batch size to use when training/updating the networks.
        ts_per_rollout: int
            A soft limit on the number of timesteps
            to allow per rollout (can span multiple
            episodes). Note that our actual timestep
            count can exceed this limit, but we won't
            start any new episodes once it has.
        gamma: float
            The 'gamma' value for calculating
            advantages and discounting rewards
            when normalizing them.
        epochs_per_iter: int
            'Epoch' is used loosely and with a variety
            of meanings in RL. In this case, a single
            epoch is a single update of all networks.
            epochs_per_iter is the number of updates
            to perform after a single rollout (which
            may contain multiple episodes).
        ext_reward_weight: float
            An optional weight for the extrinsic
            reward.
        normalize_adv: bool
            Should we normalize the advantages? This
            occurs at the minibatch level.
        normalize_obs: bool
            Should we normalize the observations?
        normalize_rewards: bool
            Should we normalize the rewards?
        normalize_values: bool
            Should we normalize the "values" that our
            critic calculates loss against?
        obs_clip: tuple or None
            Disabled if None. Otherwise, this should
            be a tuple containing a clip range for
            the observation space as (min, max).
        reward_clip: tuple or None
            Disabled if None. Otherwise, this should
            be a tuple containing a clip range for
            the reward as (min, max).
        recalc_advantages: bool
            Should we recalculate the advantages between epochs?
        render: bool
            Should we render the environment while training?
        frame_pause: float
            If render is True, sleep frame_pause seconds between renderings.
        load_state: bool
            Should we load a saved state?
        state_path: str
            The path to save/load our state.
        pretrained_policies: dict or str
            Either a string indicating the state path to load all policies from
            or a dictionary mapping policy ids to specific policy save directories.
            The saved policies must have the same structure as the policies
            defined for this training.
            dict example:
                {'policy_a' : '/foo/my-game/adversary-policy/latest', 
                 'policy_b' : '/foo/my-game-2/agent-policy/100'}"

            str example:
                '/foo/my-game/'
        env_state: str or None
            An optional path to load environment state from. This is useful
            when loading pre-trained policies.
        freeze_policies: list
            A list of policies to "freeze" the weights of. These policies will
            not be further trained and will merely act in the environment.
        freeze_scheduler: FreezeCyclingScheduler
            An optional scheduler to be used for "freeze cycling".
        checkpoint_every: int
            How often should we checkpoint? By default, we always
            save the latest model every iteration, but it overwrites
            itself. Checkpoints allow us to load previous saves.
        save_train_scores: bool
            If True, the extrinsic reward averages
            for each policy are saved every iteration.
        save_ep_scores: bool
            If True, the final episode scores (extrinsic reward averages)
            for eac policy are saved every iteration that they exist.
        save_avg_ep_len: bool
            If True, the average episode length will be saved as
            a curve for plotting.
        save_running_time: bool
            If True, the running time will be saved out as a curve every
            iteration.
        save_bs_info: bool
            If True, min, max, and avg of the boostrap value will be saved out
            as a curve every iteration.
        pickle_class: bool
            When enabled, the entire PPO class will
            be pickled and saved into the output
            directory after it's been initialized.
        soft_resets: bool
            Use "soft resets" during rollouts. This can be a bool or an
            instance of LinearStepScheduler.
        obs_augment: bool
            This is a funny option that can only be
            enabled with environments that have a
            "observation_augment" method defined.
            When enabled, this method will be used to
            augment observations into batches of
            observations that all require the same
            treatment (a single action).
        test_mode: bool
            Most of this class is not used for
            testing, but some of its attributes are.
            Setting this to True will enable test
            mode.
        force_gc: bool
            Force garbage collection? This will slow down computations,
            but it can help alleviate memory issues.
        policy_tag: str
            An optional tag to use when loading previously saved policies.
            This parameter is ignored when pretrained_policies is set.
        verbose: bool
            Enable verbosity?
        """
        set_torch_threads()

        #
        # We want each processor on each rank to collect ts_per_rollout
        # timesteps.
        #
        ts_per_rollout = num_procs * ts_per_rollout * envs_per_proc
        ts_per_rollout = int(ts_per_rollout / num_procs)

        #
        # Create our policies.
        #
        self.policies = {}
        self.have_agent_grouping           = False
        self.have_policy_step_constraints  = False
        self.have_policy_reset_constraints = False

        for policy_id in policy_settings:
            settings = policy_settings[policy_id]

            #
            # Pass our verbosity flag to the policy classes.
            #
            settings[4]["verbose"] = verbose

            self.policies[policy_id] = \
                generate_policy(
                    envs_per_proc            = envs_per_proc,
                    policy_name              = str(policy_id),
                    policy_class             = settings[0],
                    actor_observation_space  = settings[1],
                    critic_observation_space = settings[2],
                    action_space             = settings[3],
                    test_mode                = test_mode,
                    **settings[4])

            if self.policies[policy_id].agent_grouping:
                self.have_agent_grouping = True

            if self.policies[policy_id].have_step_constraints:
                self.have_policy_step_constraints = True

            if self.policies[policy_id].have_reset_constraints:
                self.have_policy_reset_constraints = True

        #
        # Create our environment instances.
        #
        self.env = wrap_environment(
            env_generator     = env_generator,
            policy_mapping_fn = policy_mapping_fn,
            policies          = self.policies,
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
        if (self.env.has_wrapper(RewardNormalizer) or
            self.env.has_wrapper(ObservationNormalizer)):
            self.save_env_info = True

        #
        # Register our agents with our policies.
        #
        for agent_id in self.env.agent_ids:
            policy_id = policy_mapping_fn(agent_id)
            self.policies[policy_id].register_agent(agent_id)

        #
        # When we toggle test mode on/off, we need to make sure to also
        # toggle this flag for any modules that depend on it.
        #
        self.test_mode_dependencies = [self.env]
        self.pickle_safe_test_mode_dependencies = []

        #
        # Establish some class variables.
        #
        self.device              = torch.device(device)
        self.state_path          = state_path
        self.render              = render
        self.frame_pause         = frame_pause
        self.ext_reward_weight   = ext_reward_weight
        self.max_ts_per_ep       = max_ts_per_ep
        self.batch_size          = batch_size
        self.ts_per_rollout      = ts_per_rollout
        self.gamma               = gamma
        self.epochs_per_iter     = epochs_per_iter
        self.prev_top_window     = -np.finfo(np.float32).max
        self.normalize_adv       = normalize_adv
        self.normalize_rewards   = normalize_rewards
        self.normalize_obs       = normalize_obs
        self.normalize_values    = normalize_values
        self.recalc_advantages   = recalc_advantages
        self.obs_augment         = obs_augment
        self.test_mode           = test_mode
        self.actor_obs_shape     = self.env.observation_space.shape
        self.policy_mapping_fn   = policy_mapping_fn
        self.envs_per_proc       = envs_per_proc
        self.verbose             = verbose
        self.force_gc            = force_gc
        self.checkpoint_every    = checkpoint_every
        self.save_train_scores   = save_train_scores
        self.save_ep_scores      = save_ep_scores
        self.save_avg_ep_len     = save_avg_ep_len
        self.save_running_time   = save_running_time
        self.save_bs_info        = save_bs_info

        self.env_info_path       = os.path.join(self.state_path, "env_info")
        os.makedirs(self.env_info_path, exist_ok=True)

        rank_print("Using device: {}".format(self.device))
        rank_print("Number of processors: {}".format(num_procs))
        rank_print("Number of environments per processor: {}".format(envs_per_proc))

        if not test_mode:
            rank_print("ts_per_rollout per rank: ~{}".format(self.ts_per_rollout))

        if callable(soft_resets):
            if type(soft_resets) != LinearStepScheduler:
                msg  = "ERROR: soft_resets must be of type bool or "
                msg += f"{LinearStepScheduler} but received "
                msg += f"{type(soft_resets)}"
                rank_print(msg)
                comm.Abort()

            self.soft_resets = soft_resets

        else:
            self.soft_resets = CallableValue(soft_resets)

        #
        # Create a dictionary to track the status of training.
        # These entries can be global, agent specific, or
        # policy specific.
        #
        max_int = np.iinfo(np.int32).max

        self.status_dict = OrderedDict()
        self.status_dict["global status"] = OrderedDict()
        self.status_dict["global status"]["iteration"]         = 0
        self.status_dict["global status"]["rollout time"]      = 0
        self.status_dict["global status"]["train time"]        = 0
        self.status_dict["global status"]["running time"]      = 0
        self.status_dict["global status"]["timesteps"]         = 0
        self.status_dict["global status"]["total episodes"]    = 0
        self.status_dict["global status"]["longest episode"]   = 0
        self.status_dict["global status"]["shortest episode"]  = max_int
        self.status_dict["global status"]["average episode"]   = 0

        for policy_id in self.policies:
            policy = self.policies[policy_id]

            self.status_dict[policy_id] = OrderedDict()
            self.status_dict[policy_id]["score avg"]            = 0
            self.status_dict[policy_id]["natural score avg"]    = 0
            self.status_dict[policy_id]["top score"]            = -max_int
            self.status_dict[policy_id]["weighted entropy"]     = 0
            self.status_dict[policy_id]["actor loss"]           = 0
            self.status_dict[policy_id]["critic loss"]          = 0
            self.status_dict[policy_id]["kl avg"]               = 0
            self.status_dict[policy_id]["natural reward range"] = (max_int, -max_int)
            self.status_dict[policy_id]["top natural reward"]   = -max_int
            self.status_dict[policy_id]["reward range"]         = (max_int, -max_int)
            self.status_dict[policy_id]["bootstrap range"]      = (max_int, -max_int)
            self.status_dict[policy_id]["bootstrap avg"]        = "N/A"
            self.status_dict[policy_id]["obs range"]            = (max_int, -max_int)
            self.status_dict[policy_id]["frozen"]               = False

        #
        # Value normalization is discussed in multiple papers, so I'm not
        # going to reference one in particular. In general, the idea is
        # to normalize the targets of the critic network using a running
        # average. The output of the critic then needs to be de-normalized
        # for calcultaing advantages. We track separate normalizers for
        # each policy.
        #
        # TODO: should we move the value normalizers to the policies?
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

        self.test_mode_dependencies.append(self.policies)
        self.pickle_safe_test_mode_dependencies.append(self.policies)

        for policy_id in self.policies:
            if self.policies[policy_id].enable_icm:
                self.status_dict[policy_id]["icm loss"] = 0
                self.status_dict[policy_id]["intrinsic score avg"] = 0
                self.status_dict[policy_id]["intr reward range"] = (max_int, -max_int)

        if load_state:
            if not os.path.exists(os.path.join(state_path, "state_0.pickle")):
                msg  = "WARNING: {state_path} does not contained saved state. "
                rank_print(msg)
                comm.Abort()
            else:
                rank_print("Loading status and environment state from {}".format(state_path))

                #
                # Let's ensure backwards compatibility with previous commits.
                #
                tmp_status_dict = self.load_status(state_path)

                for key in tmp_status_dict:
                    if key in self.status_dict:
                        self.status_dict[key] = tmp_status_dict[key]

                if env_state is None:
                    rank_print(f"Loading environment state from {env_state}")
                    self.load_env_info(self.env_info_path, tag=policy_tag)

        if env_state is not None:
            rank_print(f"Loading environment state from {env_state}")
            self.load_env_info(env_state, tag="")

        if not os.path.exists(state_path) and rank == 0:
            os.makedirs(state_path)

        self.curve_path       = os.path.join(state_path, "curves")
        self.train_score_path = os.path.join(self.curve_path, "scores")
        self.ep_score_path    = os.path.join(self.curve_path, "episode_scores")
        self.ep_len_path      = os.path.join(self.curve_path, "episode_length")
        self.runtime_path     = os.path.join(self.curve_path, "runtime")
        self.bs_min_path      = os.path.join(self.curve_path, "bs_min")
        self.bs_max_path      = os.path.join(self.curve_path, "bs_max")
        self.bs_avg_path      = os.path.join(self.curve_path, "bs_avg")

        if self.save_train_scores and rank == 0:
            if not os.path.exists(self.train_score_path):
                os.makedirs(self.train_score_path)

        if self.save_ep_scores and rank == 0:
            if not os.path.exists(self.ep_score_path):
                os.makedirs(self.ep_score_path)

        if self.save_avg_ep_len and rank == 0:
            if not os.path.exists(self.ep_len_path):
                os.makedirs(self.ep_len_path)

        if self.save_running_time and rank == 0:
            if not os.path.exists(self.runtime_path):
                os.makedirs(self.runtime_path)

        if self.save_bs_info and rank == 0:
            if not os.path.exists(self.bs_min_path):
                os.makedirs(self.bs_min_path)

            if not os.path.exists(self.bs_max_path):
                os.makedirs(self.bs_max_path)

            if not os.path.exists(self.bs_avg_path):
                os.makedirs(self.bs_avg_path)

        comm.barrier()

        for policy_id in self.policies:
            policy = self.policies[policy_id]

            #
            # Finalized needs to be called after the status dictionary
            # has been loaded so that we can update our learning
            # rates and various weights (not model weights).
            #
            self.policies[policy_id].finalize(self.status_dict, self.device)
            self.policies[policy_id].seed(random_seed)

            self.status_dict[policy_id]["lr"] = policy.lr()
            self.status_dict[policy_id]["entropy weight"] = \
                policy.entropy_weight()

            if self.policies[policy_id].enable_icm:
                self.status_dict[policy_id]["icm lr"] = \
                    policy.icm_lr()
                self.status_dict[policy_id]["intr reward weight"] = \
                    policy.intr_reward_weight()

        pretrained_is_direct = True
        if type(pretrained_policies) == str:
            pretrained_path      = pretrained_policies
            pretrained_policies  = {}
            pretrained_is_direct = False

            for policy_id in self.policies:
                pretrained_policies[policy_id] = pretrained_path

        elif type(pretrained_policies) != dict:
            msg  = "ERROR: pretrained_policies must be of type str or "
            msg += "dict but received {type(pretrained_policies)}"
            rank_print(msg)
            comm.Abort()

        #
        # Load the policies after they've been finalized.
        #
        if load_state or len(pretrained_policies) > 0:

            #
            # First, load any policies that are local/not pretrained.
            #
            if load_state and os.path.exists(state_path):
                for policy_id in self.policies:
                    if policy_id not in pretrained_policies:
                        rank_print(f"Loading policy {policy_id} from {state_path} with tag {policy_tag}")
                        self.load_policy(policy_id, state_path, tag=policy_tag)

            #
            # Second, load our pretrained policies.
            #
            for policy_id in pretrained_policies:
                if policy_id not in self.policies:
                    msg  = "ERROR: pre-trained policy {policy_id} is not "
                    msg += "registered in this environment."
                    rank_print(msg)
                    comm.Abort()

                pretrained_path = pretrained_policies[policy_id]
                rank_print(f"Loading pre-trained policy {policy_id} from {pretrained_path}")

                if pretrained_is_direct:
                    self.direct_load_policy(policy_id, pretrained_path)
                else:
                    self.load_policy(policy_id, pretrained_path, tag=policy_tag)

        for policy_id in freeze_policies:
            if policy_id not in self.policies:
                msg  = f"ERROR: freeze policy {policy_id} is not registered "
                msg += f"with this environment."
                rank_print(msg) 
                comm.Abort()

            self.policies[policy_id].freeze()

        if freeze_scheduler is None:
            self.freeze_scheduler = CallableValue(None)
        else:
            if not issubclass(type(freeze_scheduler), FreezeCyclingScheduler):
                msg  = "ERROR: freeze_scheduler must be a subclass of "
                msg += f"FreezeCyclingScheduler but received type "
                msg += f"{type(freeze_scheduler)}"
                rank_print(msg)
                comm.Abort()

            self.freeze_scheduler = freeze_scheduler

        self.env.finalize(self.status_dict)
        self.soft_resets.finalize(self.status_dict)
        self.freeze_scheduler.finalize(self.state_path, self.status_dict, self.policies)
        self.freeze_scheduler.load_info()

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

        if self.verbose and rank == 0:
            rank_print(f"\nPolicy info:")
            rank_print(f"  training {len(self.policies)} policies.")
            for policy_id in self.policies:
                policy = self.policies[policy_id]
                sp     = "  "
                rank_print(f"\n  {policy.name} policy:")
                rank_print(f"{sp}action space: {policy.action_space}")
                rank_print(f"{sp}actor obs space: {policy.actor_obs_space}")
                rank_print(f"{sp}critic obs space: {policy.critic_obs_space}")
                rank_print(f"{sp}agent ids: {policy.agent_ids}")

        comm.barrier()

        self.full_len_ep_scores = EpisodeScores(
            self.env.get_batch_size(),
            np.array(list(self.policies.keys())))

    def get_policy_batches(self, obs, component):
        """
        This method will take all of the observations from a step
        and compress them into numpy array batches. This allows for
        much faster inference during rollouts.

        Parameters:
        -----------
        obs: array-like
            The observations to create batches from. This
            should be a dictionary mapping agent ids to
            their observations.
        component: str
            The network component these observations are
            associated with. This can be set to "actor"
            or "critic".

        Returns:
        --------
        tuple:
            The first element is dictionary mapping policy ids to arrays of
            agent ids in the order they appear in the batches. Next is
            a dictionary mapping policy ids to batches of agent observations.
            These batches have shape (num_agents, envs_per_proc, obs_shape).
        """
        assert component in ["actor", "critic"]

        policy_batches   = {}
        agent_counts     = {}
        policy_agent_ids = {}

        #
        # First, let's populate our "agent_counts" dictionary, which maps
        # policy ids to number of agents from these observations that
        # correspond to the policy.
        #
        for a_id in obs:
            policy_id = self.policy_mapping_fn(a_id)

            if policy_id not in agent_counts:
                agent_counts[policy_id] = 0

            agent_counts[policy_id] += 1

        #
        # Next, combine observations from all agents that share policies.
        # We'll add these to our dictionary mapping policy ids to batches.
        #
        for policy_id in agent_counts:

            if (agent_counts[policy_id] !=
                len(self.policies[policy_id].agent_ids)):

                msg  = f"ERROR: expected obs of policy {policy_id} to have "
                msg += f"{len(self.policies[policy_id].agent_ids)} agents "
                msg += f"but recieved {agent_counts[policy_id]}!"
                rank_print(msg)
                comm.Abort()

            if component == "actor":
                policy_shape = self.policies[policy_id].actor_obs_space.shape
            elif component == "critic":
                policy_shape = self.policies[policy_id].critic_obs_space.shape

            batch_shape = (agent_counts[policy_id], self.envs_per_proc) +\
                policy_shape

            if component == "actor":
                policy_batches[policy_id] = np.zeros(batch_shape).astype(
                    self.policies[policy_id].actor_obs_space.dtype)
            elif component == "critic":
                policy_batches[policy_id] = np.zeros(batch_shape).astype(
                    self.policies[policy_id].critic_obs_space.dtype)

            agent_ids = self.policies[policy_id].agent_ids.copy()

            policy_agent_ids[policy_id] = agent_ids

            #
            # NOTE: this enforces agent ordering which is needed for
            # order-sensitive algorithms like MAT.
            #
            for a_idx, a_id in enumerate(policy_agent_ids[policy_id]):
                policy_batches[policy_id][a_idx] = obs[a_id]

        return policy_agent_ids, policy_batches

    def get_rollout_actions(self, obs):
        """
        Given a dictionary mapping agent ids to observations,
        generate an dictionary of actions from our policy.

        Parameters:
        -----------
        obs: dict
            A dictionary mapping agent ids to observations.
        ac_component: string
            Either 'actor' or 'critic'. This should match which component the
            observations are associated with. In most cases, this is
            the actor, but cases like MAT only use the critic observations.

        Returns:
        --------
        tuple:
            A tuple of the form (raw_actions, actions, log_probs).
            'actions' have potentially been altered for the environment,
            but 'raw_actions' are guaranteed to be unaltered.
        """
        raw_actions = {}
        actions     = {}
        log_probs   = {}

        #
        # Performing inference on each agent individually is VERY slow.
        # Instead, we can batch all shared policy observations.
        # Also, some algorithms (like MAT) need agent's to be batched
        # together.
        #
        policy_agent_ids, policy_batches = self.get_policy_batches(obs, "actor")

        for policy_id in policy_batches:
            batch_obs  = policy_batches[policy_id]
            num_agents = batch_obs.shape[0]

            if not self.policies[policy_id].agent_grouping:
                batch_obs  = batch_obs.reshape((-1,) + \
                    self.policies[policy_id].actor_obs_space.shape)

            batch_raw_actions, batch_actions, batch_log_probs = \
                self.policies[policy_id].get_rollout_actions(batch_obs)

            #
            # We now need to reverse our batching to get actions of
            # shape (num_agents, num_envs, actions).
            #
            actions_shape = (num_agents, self.envs_per_proc) + \
                self.policies[policy_id].action_space.shape

            batch_raw_actions = batch_raw_actions.reshape(actions_shape)
            batch_actions     = batch_actions.reshape(actions_shape)
            batch_log_probs   = batch_log_probs.reshape(num_agents,
                self.envs_per_proc, -1)

            for a_idx, a_id in enumerate(policy_agent_ids[policy_id]):
                raw_actions[a_id] = batch_raw_actions[a_idx]
                actions[a_id]     = batch_actions[a_idx]
                log_probs[a_id]   = batch_log_probs[a_idx]

        return raw_actions, actions, log_probs

    def get_rollout_actions_from_aug_obs(self, obs):
        """
        Given a dictionary mapping agent ids to augmented
        batches of observations,
        generate an dictionary of actions from our policy.

        Parameters:
        -----------
        obs: dict
            A dictionary mapping agent ids to observations.

        Returns:
        --------
        tuple:
            A tuple of the form (raw_actions, actions, log_probs).
            'actions' have potentially been altered for the environment,
            but 'raw_actions' are guaranteed to be unaltered.
        """
        raw_actions = {}
        actions     = {}
        log_probs   = {}

        #TODO: update this to use policy batches.
        for agent_id in obs:
            policy_id = self.policy_mapping_fn(agent_id)

            obs_slice = obs[agent_id][0:1]
            raw_action, action, log_prob = \
                self.policies[policy_id].get_rollout_actions(obs_slice)

            raw_actions[agent_id] = raw_action
            actions[agent_id]     = action
            log_probs[agent_id]   = log_prob

        return raw_actions, actions, log_probs

    def get_inference_actions(self, obs, deterministic):
        """
        Get actions to be used for evaluation or inference in a
        deployment.

        Parameters:
        -----------
        obs: dict
            A dictionary mapping agent ids to observations.
        deterministic: bool
            If True, the action will always come from the highest
            probability action. Otherwise, our actions come from
            sampling the distribution.

        Returns:
        --------
        dict:
            A dictionary mapping agent ids to actions.
        """
        if self.have_agent_grouping:
            return self._get_policy_grouped_inference_actions(obs, deterministic)
        return self._get_mappo_inference_actions(obs, deterministic)

    def _get_policy_grouped_inference_actions(self, obs, deterministic):
        """
        Get actions to be used for evaluation or inference in a
        deployment. This function is specifically for use with
        policies that group their agents together (like MAT).

        Parameters:
        -----------
        obs: dict
            A dictionary mapping agent ids to observations.
        deterministic: bool
            If True, the action will always come from the highest
            probability action. Otherwise, our actions come from
            sampling the distribution.

        Returns:
        --------
        dict:
            A dictionary mapping agent ids to actions.
        """
        actions    = {}
        policy_obs = {}

        #
        # First, let's split the agents up into separate obs dictionaries
        # for each policy. The MAT policies
        #
        for agent_id in obs:
            policy_id = self.policy_mapping_fn(agent_id)

            if policy_id not in policy_obs:
                policy_obs[policy_id] = {}

            policy_obs[policy_id][agent_id] = obs[agent_id]

        actions = {}

        #
        # Next, we handle MAT policies and standard PPO policies
        # distinctly. Standard PPO policies take in a single agent's
        # observation, while MAT policies take in all of the agent's
        # observations.
        #
        for policy_id in policy_obs:
            if self.policies[policy_id].agent_grouping:
                policy_agent_ids, batch_obs = self.get_policy_batches(
                    obs            = policy_obs[policy_id],
                    component      = "actor")

                agent_ids = policy_agent_ids[policy_id]
                batch_obs = batch_obs[policy_id]

                batch_actions = \
                    self.policies[policy_id].get_inference_actions(
                        batch_obs, deterministic)

                #
                # We now need to reverse our batching and put the agents
                # back into dictionary form.
                #
                num_agents    = len(agent_ids)
                actions_shape = (num_agents,) +\
                    self.policies[policy_id].action_space.shape
                batch_actions = batch_actions.reshape(actions_shape).numpy()

                policy_actions = {}
                for a_idx, a_id in enumerate(agent_ids):
                    policy_actions[a_id] = batch_actions[a_idx]

            else:
                policy_actions = self._get_mappo_inference_actions(
                    policy_obs[policy_id], deterministic)

            actions.update(policy_actions)

        return actions

    def _get_mappo_inference_actions(self, obs, deterministic):
        """
        Get actions to be used for evaluation or inference in a
        deployment. This function is specifically for use with
        standard MAPPO like policies that don't use any agent
        grouping.

        Parameters:
        -----------
        obs: dict
            A dictionary mapping agent ids to observations.
        deterministic: bool
            If True, the action will always come from the highest
            probability action. Otherwise, our actions come from
            sampling the distribution.

        Returns:
        --------
        dict:
            A dictionary mapping agent ids to actions.
        """
        actions = {}
        for agent_id in obs: 

            obs[agent_id] = np.expand_dims(obs[agent_id], 0)
            policy_id     = self.policy_mapping_fn(agent_id)

            agent_action = self.policies[policy_id].get_inference_actions(
                obs[agent_id], deterministic)

            actions[agent_id] = agent_action.numpy()

        return actions

    def get_policy_values(self, obs):
        """
        Given a dictionary mapping agent ids to observations,
        construct a dictionary mapping agent ids to values
        predicted by the policy critics.

        Parameters:
        -----------
        obs: dict
            A dictionary mapping agent ids to observations.

        Returns:
        --------
        dict:
            A dictionary mapping agent ids to critic values.
        """
        values = {}

        #
        # Performing inference on each agent individually is VERY slow.
        # Instead, we can batch all shared policy observations.
        #
        policy_agent_ids, policy_batches = self.get_policy_batches(obs, "critic")
        policy_batches = self.np_dict_to_tensor_dict(policy_batches)

        for policy_id in policy_batches:
            batch_obs  = policy_batches[policy_id]
            num_agents = batch_obs.shape[0]

            if self.policies[policy_id].agent_grouping:
                batch_obs = batch_obs.swapaxes(0, 1)
            else:
                batch_obs  = batch_obs.reshape((-1,) + \
                    self.policies[policy_id].critic_obs_space.shape)

            batch_values = self.policies[policy_id].get_critic_values(batch_obs)

            if self.policies[policy_id].agent_grouping:
                batch_obs = batch_obs.swapaxes(0, 1)

            batch_values = batch_values.reshape((num_agents, -1))

            for b_idx, a_id in enumerate(policy_agent_ids[policy_id]):
                values[a_id] = batch_values[b_idx]

        return values

    def get_natural_reward(self, info):
        """
        Given an info dictionary, construct a dictionary mapping
        agent ids to their natural rewards.

        Parameters:
        -----------
        info: dict
            The info dictionary. Each element is a sub-dictionary
            mapping agent ids to their info.

        Returns:
        --------
        tuple:
            A tuple of form (have_natural_rewards, natural_rewards) s.t.
            the first index is a boolean signifying whether or not natural
            rewards were found, and the second index contains a dictionary
            mapping agent ids to their natural rewards.
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

    def get_detached_dict(self, attached):
        """
        Given a dictionary mapping agent ids to torch
        tensors, create a replica of this dictionary
        containing detached numpy arrays.

        Parameters:
        -----------
        attached: dict
            A dictionary mapping agent ids to torch tensors.

        Returns:
        --------
        dict:
            A replication of "attached" that maps to numpy arrays.
        """
        detached = {}

        for agent_id in attached:
            if torch.is_tensor(attached[agent_id]):
                detached[agent_id] = \
                    attached[agent_id].detach().cpu().numpy()
            else:
                detached[agent_id] = \
                    attached[agent_id]

        return detached

    def get_denormalized_values(self, values):
        """
        Given a dictionary mapping agent ids to critic values,
        return a replica of this dictionary containing de-normalized
        values.

        Parameters:
        -----------
        values: dict
            A dictionary mapping agnet ids to values.

        Returns:
        --------
        dict:
            A replica of "values" mapping to de-normalized values.
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
        Given a dictionary mapping agent ids to critic values,
        return a replica of this dictionary containing normalized
        values.

        Parameters:
        -----------
        values: dict
            A dictionary mapping agnet ids to values.

        Returns:
        --------
        dict:
            A replica of "values" mapping to normalized values.
        """
        norm_values = {}

        for agent_id in values:
            policy_id = self.policy_mapping_fn(agent_id)
            value     = values[agent_id]
            value     = self.value_normalizers[policy_id].normalize(value)
            norm_values[agent_id] = value

        return norm_values

    def np_dict_to_tensor_dict(self, numpy_dict):
        """
        Given a dictionary mapping agent ids to numpy arrays,
        return a replicat of this dictionary mapping to torch
        tensors.

        Parameters:
        -----------
        numpy_dict: dict
            A dictionary mapping agent ids to numpy arrays.

        Returns:
        --------
        dict:
            A replica of "numpy_dict" that maps to torch tensors.
        """
        tensor_dict = {}

        for agent_id in numpy_dict:
            tensor_dict[agent_id] = torch.tensor(numpy_dict[agent_id],
                dtype=torch.float32).to(self.device)

        return tensor_dict

    def apply_intrinsic_rewards(self,
                                ext_rewards,
                                prev_obs,
                                obs,
                                actions):
        """
        Apply intrinsic rewards to our extrinsic rewards when using
        ICM.

        Parameters:
        -----------
        ext_rewards: dict
            The extrinsic rewards dictionary.
        prev_obs: dict
            The previous observation dictionary.
        obs: dict
            The current observation dictionary.
        actions: dict
            The actions dictionary.

        Returns:
        --------
        tuple:
            A tuple of form (rewards, intr_rewards) s.t. "rewards" is
            an updated version of the input rewards that have the intrinsic
            rewards applied, and "intr_rewards" is a dictionary containing
            the intrinsic rewards alone.
        """
        intr_rewards = {}
        rewards      = {}
        remaining_policies = len(self.policies)

        for policy_id in self.policies:
        
            if (self.policies[policy_id].enable_icm and
                self.policies[policy_id].agent_shared_icm):

                shared_intr_reward = self.policies[policy_id].get_agent_shared_intrinsic_rewards(
                    prev_obs,
                    obs,
                    actions)

                for agent_id in self.policies[policy_id].agent_ids:
                    intr_rewards[agent_id] = shared_intr_reward
                    rewards[agent_id]      = ext_rewards[agent_id] + shared_intr_reward

                remaining_policies -= 1

        #
        #
        #
        if remaining_policies > 0:
            for agent_id in obs:
                policy_id = self.policy_mapping_fn(agent_id)

                if (self.policies[policy_id].enable_icm and not
                    self.policies[policy_id].agent_shared_icm):

                    intr_rewards[agent_id] = \
                        self.policies[policy_id].get_intrinsic_reward(
                            prev_obs[agent_id],
                            obs[agent_id],
                            actions[agent_id])

                    rewards[agent_id] = ext_rewards[agent_id] + intr_rewards[agent_id]
                else:
                    rewards[agent_id] = ext_rewards[agent_id]
                    intr_rewards[agent_id] = np.zeros(1)

        return rewards, intr_rewards

    def apply_reward_weight(self,
                            rewards,
                            weight):
        """
        Apply a wieght to a reward dictionary.

        Parameters:
        -----------
        rewards: dict
            The rewards dictionary.
        weight: float
            A weight to apply to all rewards.

        Returns:
        --------
        dict:
            The input rewards dictionary after applying the weight.
        """
        for agent_id in rewards:
            rewards[agent_id] *= weight

        return rewards

    def get_terminated_envs(self,
                            terminated):
        """
        Determine which environments are terminated. Because we death mask,
        we will never be in a situation where an agent is termintaed before
        its associated environment is terminated.

        Parameters:
        -----------
        terminated: dict
            The terminated dictionary.

        Returns:
        --------
        tuple:
            A tuple of form (where_term, where_not_term), which contains
            numpy arrays determining which environments are terminated/
            not terminated.
        """
        first_id   = next(iter(terminated))
        batch_size = terminated[first_id].size
        term_envs  = np.zeros(batch_size).astype(bool)

        #
        # Because we always death mask, any agent that has a terminated
        # environment means that all agents are terminated in that same
        # environment.
        #
        where_term     = np.where(terminated[first_id])[0]
        where_not_term = np.where(~terminated[first_id])[0]

        return where_term, where_not_term

    def _tile_aug_results(self, action, raw_action, obs, log_prob):
        """
        When in-line augmentation is enabled, we need to tile
        some of our results from taking a step. The observations
        are augmented, and the actions remain the same.

        Parameters:
        -----------
        action: dict
            The action dictionary.
        raw_actio: dict
            The raw action dictionary.
        obs: dict
            The observation dictionary.
        log_prrob: dict
            The log prob dictionary.
        """
        for agent_id in obs:
            batch_size   = obs[agent_id].shape[0]

            action_shape = (batch_size,) + action[agent_id].shape[1:]

            action[agent_id] = np.tile(
                action[agent_id].flatten(), batch_size)
            action[agent_id] = action[agent_id].reshape(action_shape)

            raw_action[agent_id] = np.tile(
                raw_action[agent_id].flatten(), batch_size)
            raw_action[agent_id] = \
                raw_action[agent_id].reshape(action_shape)

            lp_shape = (batch_size,) + log_prob[agent_id].shape[1:]

            log_prob[agent_id] = np.tile(
                log_prob[agent_id].flatten(), batch_size)
            log_prob[agent_id] = log_prob[agent_id].reshape(lp_shape)

    def print_status(self):
        """
        Print out statistics from our status_dict.
        """
        rank_print("\n--------------------------------------------------------")
        rank_print("Status Report:")
        rank_print("  global status:")
        for key in self.status_dict["global status"]:
            if key in ["running time", "rollout time", "train time"]:
                pretty_time = format_seconds(self.status_dict["global status"][key])
                rank_print("    {}: {}".format(key, pretty_time))
            else:
                rank_print("    {}: {}".format(key,
                    self.status_dict["global status"][key]))

        for policy_id in self.policies:
            rank_print("  {}:".format(policy_id))
            for key in self.status_dict[policy_id]:
                rank_print("    {}: {}".format(key,
                    self.status_dict[policy_id][key]))

        rank_print("--------------------------------------------------------")

    def update_learning_rate(self):
        """
        Update the learning rate.
        """
        for policy_id in self.policies:
            self.policies[policy_id].update_learning_rate()
            self.status_dict[policy_id]["lr"] = self.policies[policy_id].lr()

            if self.policies[policy_id].enable_icm:
                self.status_dict[policy_id]["icm lr"] = \
                    self.policies[policy_id].icm_lr()
                self.status_dict[policy_id]["intr reward weight"] = \
                    self.policies[policy_id].intr_reward_weight()

    def update_entropy_weight(self):
        """
        Update the entropy weight.
        """
        for policy_id in self.policies:
            self.status_dict[policy_id]["entropy weight"] = \
                self.policies[policy_id].entropy_weight()

    def verify_truncated(self, terminated, truncated):
        """
        Make sure that our environment terminations and truncations
        are behaving as expected; only one should be True at any given
        time.

        Paramters:
        ----------
        terminated: dict
            Dictionary mapping agent ids to bool terminated flags.
        truncated: dict
            Dictionary mapping agent ids to bool truncated flags.

        Returns:
        --------
        np.ndarray
            Numpy array marking which environments were truncated.
        """
        first_agent     = next(iter(truncated))
        where_truncated = np.where(truncated[first_agent])

        for agent_id in terminated:
            #
            # This is an odd edge case. We shouldn't have both
            # types of done simultaneously...
            #
            if terminated[agent_id][where_truncated].any():
                terminated[agent_id][where_truncated] = False

                if self.verbose:
                    msg  = "WARNING: terminated and truncated were both "
                    msg += "set to True. Setting terminated to False."
                    rank_print(msg)

            msg  = "ERROR: truncation for one but not all agents in "
            msg += "an environment is not currently supported."
            assert truncated[agent_id][where_truncated].all(), msg

        return where_truncated

    def apply_policy_reset_constraints(
        self,
        obs,
        critic_obs):
        """
        Apply any policy constraints needed when resetting the environment.

        NOTE: This may alter the values returned by the environment.

        Parameters:
        -----------
        obs: dict
            Dictionary mapping agent ids to actor observations.
        critic_obs: dict
            Dictionary mapping agent ids to critic observations.
        """
        if self.have_policy_reset_constraints:
            for policy_id in self.policies:
                obs, critic_obs  = \
                    self.policies[policy_id].apply_reset_constraints(
                        obs,
                        critic_obs)

        return obs, critic_obs

    def apply_policy_step_constraints(
        self,
        obs,
        critic_obs,
        reward,
        terminated,
        truncated,
        info):
        """
        Apply any policy constraints needed when stepping through the environment.

        NOTE: This may alter the values returned by the environment.

        Parameters:
        -----------
        obs: dict
            Dictionary mapping agent ids to actor observations.
        critic_obs: dict
            Dictionary mapping agent ids to critic observations.
        reward: dict
            Dictionary mapping agent ids to rewards.
        terminated: dict
            Dictionary mapping agent ids to a termination flag.
        truncated: dict
            Dictionary mapping agent ids to a truncated flag.
        info: dict
            Dictionary mapping agent ids to info dictionaries.
        """
        if self.have_policy_step_constraints:
            for policy_id in self.policies:
                obs, critic_obs, reward, terminated, truncated, info = \
                    self.policies[policy_id].apply_step_constraints(
                        obs,
                        critic_obs,
                        reward,
                        terminated,
                        truncated,
                        info)

        return obs, critic_obs, reward, terminated, truncated, info

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
        --------
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
        total_bs         = 0
        total_rollout_ts = 0
        longest_run      = 0
        shortest_run     = self.ts_per_rollout / self.envs_per_proc
        avg_run          = self.ts_per_rollout / self.envs_per_proc

        for key in self.policies:
            self.policies[key].eval()

        #
        # TODO: soft resets might cause rollouts to start off in "traps"
        # that are impossible to escape. We might be able to handle this
        # more intelligently.
        #
        if self.soft_resets():
            initial_reset_func = self.env.soft_reset
        else:
            initial_reset_func = self.env.reset

        obs, critic_obs = self.apply_policy_reset_constraints(
            *initial_reset_func())

        env_batch_size  = self.env.get_batch_size()

        top_rollout_score       = {}
        rollout_max_reward      = {}
        rollout_min_reward      = {}
        rollout_max_nat_reward  = {}
        rollout_min_nat_reward  = {}
        rollout_max_intr_reward = {}
        rollout_min_intr_reward = {}
        rollout_max_obs         = {}
        rollout_min_obs         = {}
        ep_nat_scores           = {}
        ep_scores               = {}
        ep_intr_scores          = {}
        total_nat_scores        = {}
        total_intr_scores       = {}
        total_scores            = {}
        top_reward              = {}
        bs_min                  = {}
        bs_max                  = {}
        bs_sum                  = {}

        for policy_id in self.policies:
            top_rollout_score[policy_id]       = -np.finfo(np.float32).max
            rollout_max_reward[policy_id]      = -np.finfo(np.float32).max
            rollout_min_reward[policy_id]      = np.finfo(np.float32).max
            rollout_max_nat_reward[policy_id]  = -np.finfo(np.float32).max
            rollout_min_nat_reward[policy_id]  = np.finfo(np.float32).max
            rollout_max_intr_reward[policy_id] = -np.finfo(np.float32).max
            rollout_min_intr_reward[policy_id] = np.finfo(np.float32).max
            rollout_max_obs[policy_id]         = -np.finfo(np.float32).max
            rollout_min_obs[policy_id]         = np.finfo(np.float32).max
            ep_nat_scores[policy_id]           = np.zeros((env_batch_size, 1))
            ep_intr_scores[policy_id]          = np.zeros((env_batch_size, 1))
            ep_scores[policy_id]               = np.zeros((env_batch_size, 1))
            total_nat_scores[policy_id]        = np.zeros((env_batch_size, 1))
            total_intr_scores[policy_id]       = np.zeros((env_batch_size, 1))
            total_scores[policy_id]            = np.zeros((env_batch_size, 1))
            top_reward[policy_id]              = -np.finfo(np.float32).max
            bs_min[policy_id]                  = np.finfo(np.float32).max
            bs_max[policy_id]                  = -np.finfo(np.float32).max
            bs_sum[policy_id]                  = np.zeros((env_batch_size,))

        episode_lengths = np.zeros(env_batch_size).astype(np.int32)
        ep_ts           = np.zeros(env_batch_size).astype(np.int32)

        for policy_id in self.policies:
            self.policies[policy_id].initialize_episodes(
                env_batch_size, self.status_dict)

            #
            # NOTE: The MAT paper proposes shuffling agents once per iteration.
            # I didn't actually see this happening in their code, but I find
            # that it does improve training quality a bit.
            #
            if self.policies[policy_id].agent_grouping:
                self.policies[policy_id].shuffle_agent_ids()

        while total_rollout_ts < self.ts_per_rollout:

            ep_ts += 1

            if self.render:
                self.env.render(frame_pause = self.frame_pause)

            total_rollout_ts += env_batch_size
            episode_lengths  += 1

            if self.obs_augment:
                raw_action, action, log_prob = \
                    self.get_rollout_actions_from_aug_obs(obs)
            else:
                raw_action, action, log_prob = \
                    self.get_rollout_actions(obs)

            value = self.get_policy_values(critic_obs)

            if self.normalize_values:
                value = self.get_denormalized_values(value)

            #
            # Note that we have critic observations as well as local.
            # When we're learning from a multi-agent environment, we
            # feed a "global state" to the critic. This is called
            # Centralized Training Decentralized Execution (CTDE).
            # arXiv:2006.07869v4
            #
            prev_obs        = deepcopy(obs)
            prev_critic_obs = deepcopy(critic_obs)

            #
            # The returned objects are dictionaries mapping agent ids
            # to np arrays. Each element of the numpy array represents
            # the results from a single environment.
            #
            obs, critic_obs, ext_reward, terminated, truncated, info = \
                self.apply_policy_step_constraints(*self.env.step(action))

            #
            # Because we always death mask, any environment that's done for
            # one agent is done for them all.
            #
            where_truncated = self.verify_truncated(terminated, truncated)[0]
            have_truncated  = where_truncated.size > 0

            #
            # In the observational augment case, our action is a single action,
            # but our return values are all batches. We need to tile the
            # actions into batches as well.
            #
            if self.obs_augment:
                self._tile_aug_results(self, action, raw_action, obs, log_prob)

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
            reward, intr_reward = self.apply_intrinsic_rewards(
                ext_reward,
                prev_obs,
                obs,
                action)

            ep_obs = deepcopy(obs)

            where_term, where_not_term = self.get_terminated_envs(terminated)
            term_count = where_term.size

            for agent_id in action:
                if term_count > 0:
                    #
                    # If an environment is terminal, all agents that are
                    # in that environment are in a terminal state.
                    #
                    for term_idx in where_term:
                        ep_obs[agent_id][term_idx] = \
                            info[agent_id][term_idx]["terminal observation"]

                policy_id = self.policy_mapping_fn(agent_id)

                self.policies[policy_id].add_episode_info(
                    agent_id             = agent_id,
                    critic_observations  = prev_critic_obs[agent_id],
                    observations         = prev_obs[agent_id],
                    next_observations    = ep_obs[agent_id],
                    raw_actions          = raw_action[agent_id],
                    actions              = action[agent_id],
                    values               = value[agent_id],
                    log_probs            = log_prob[agent_id],
                    rewards              = reward[agent_id],
                    where_done           = where_term)

                rollout_max_reward[policy_id] = \
                    max(rollout_max_reward[policy_id],
                        reward[agent_id].max())

                rollout_min_reward[policy_id] = \
                    min(rollout_min_reward[policy_id],
                        reward[agent_id].min())

                rollout_max_nat_reward[policy_id] = \
                    max(rollout_max_nat_reward[policy_id],
                        natural_reward[agent_id].max())

                rollout_min_nat_reward[policy_id] = \
                    min(rollout_min_nat_reward[policy_id],
                        natural_reward[agent_id].min())

                rollout_max_intr_reward[policy_id] = \
                    max(rollout_max_intr_reward[policy_id],
                        intr_reward[agent_id].max())

                rollout_min_intr_reward[policy_id] = \
                    min(rollout_min_intr_reward[policy_id],
                        intr_reward[agent_id].min())

                rollout_max_obs[policy_id]    = \
                    max(rollout_max_obs[policy_id], obs[agent_id].max())

                rollout_min_obs[policy_id]    = \
                    min(rollout_min_obs[policy_id], obs[agent_id].min())

                ep_scores[policy_id]       += reward[agent_id]
                ep_nat_scores[policy_id]   += natural_reward[agent_id]
                ep_intr_scores[policy_id]  += intr_reward[agent_id]

                self.full_len_ep_scores.add_scores(policy_id,
                    natural_reward[agent_id])

                top_reward[policy_id] = max(top_reward[policy_id],
                    natural_reward[agent_id].max())

            #
            # Episode end cases.
            #  1. An episode has reached a terminated state.
            #  2. An episode has reached the maximum allowable timesteps.
            #  3. An episode has reached a truncated state.
            #
            # Case 1.
            # We handle any episodes that have reached a terminal state.
            # In these cases, the environment cannot proceed any further.
            #
            if term_count > 0:
                #
                # Every agent has at least one terminated environment, which
                # means that every policy has at least one terminated
                # enviornment.
                #
                for agent_id in terminated:
                    policy_id = self.policy_mapping_fn(agent_id)

                    self.policies[policy_id].end_episodes(
                        agent_id        = agent_id,
                        env_idxs        = where_term,
                        episode_lengths = episode_lengths,
                        terminal        = np.ones(term_count).astype(bool),
                        ending_values   = np.zeros(term_count),
                        ending_rewards  = np.zeros(term_count))

                    top_rollout_score[policy_id] = \
                        max(top_rollout_score[policy_id],
                        ep_nat_scores[policy_id][where_term].max())

                    total_nat_scores[policy_id][where_term] += \
                        ep_nat_scores[policy_id][where_term]

                    self.full_len_ep_scores.end_episodes(policy_id, where_term)

                    if self.policies[policy_id].enable_icm:
                        total_intr_scores[policy_id][where_term] += \
                            ep_intr_scores[policy_id][where_term]

                    total_scores[policy_id][where_term] += \
                        ep_scores[policy_id][where_term]

                    ep_scores[policy_id][where_term]      = 0
                    ep_nat_scores[policy_id][where_term]  = 0
                    ep_intr_scores[policy_id][where_term] = 0

                longest_run = max(longest_run,
                    episode_lengths[where_term].max())

                shortest_run = min(shortest_run,
                    episode_lengths[where_term].min())

                avg_run = episode_lengths[where_term].mean()

                episode_lengths[where_term]    = 0
                total_episodes                += term_count
                ep_ts[where_term]              = 0

            #
            # Cases 2 and 3.
            # We handle episodes that have reached or exceeded the maximum
            # number of timesteps allowed, but they haven't yet reached a
            # terminal state. This is also very similar to reaching
            # an environment triggered truncated state, so we handle
            # them at the same time (identically).
            # Since the environment can continue, we can take this into
            # consideration when calculating the reward.
            #
            ep_max_reached = ((ep_ts == self.max_ts_per_ep).any() and
                where_not_term.size > 0)

            if (ep_max_reached or
                total_rollout_ts >= self.ts_per_rollout or
                have_truncated):

                if total_rollout_ts >= self.ts_per_rollout:
                    where_maxed = np.arange(env_batch_size)
                else:
                    where_maxed = np.where(ep_ts >= self.max_ts_per_ep)[0]

                where_maxed = np.setdiff1d(where_maxed, where_term)
                where_maxed = np.concatenate((where_maxed, where_truncated))
                where_maxed = np.unique(where_maxed)

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
                #        works best.
                #     2. If we can clone the environment, we can take
                #        an extra step with the clone to get the
                #        intrinsic reward, and we can decide what to
                #        do with this. I've tested this out as well,
                #        but I haven't seen much advantage.
                #
                # We can hand wavily calcluate a "surprise" by taking
                # the difference between the average intrinsic reward
                # and the one we get. Adding that to the critic's
                # output can act as an extra surprise bonus.
                #
                maxed_count = where_maxed.size

                if maxed_count > 0:

                    for agent_id in next_reward:
                        policy_id = self.policy_mapping_fn(agent_id)

                        bs_min[policy_id] = min(bs_min[policy_id],
                            float(next_reward[agent_id].min()))

                        bs_max[policy_id] = max(bs_max[policy_id],
                            float(next_reward[agent_id].max()))

                        bs_sum[policy_id] += next_reward[agent_id]
                        total_bs += 1

                        if self.policies[policy_id].enable_icm:
                            ism = self.status_dict[policy_id]["intrinsic score avg"]
                            surprise = intr_reward[agent_id][where_maxed] - ism

                            next_reward[agent_id] += surprise.flatten()
                        
                        self.policies[policy_id].end_episodes(
                            agent_id        = agent_id,
                            env_idxs        = where_maxed,
                            episode_lengths = episode_lengths,
                            terminal        = np.zeros(maxed_count).astype(bool),
                            ending_values   = next_value[agent_id],
                            ending_rewards  = next_reward[agent_id])

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

                        total_nat_scores[policy_id] += \
                            ep_nat_scores[policy_id]

                        total_scores[policy_id] += \
                            ep_scores[policy_id]

                        if self.policies[policy_id].enable_icm:
                            total_intr_scores[policy_id] += \
                                ep_intr_scores[policy_id]

                        if where_not_term.size > 0:
                            top_rollout_score[policy_id] = \
                                max(top_rollout_score[policy_id],
                                ep_nat_scores[policy_id][where_not_term].max())

                ep_ts[where_maxed] = 0

            longest_run = max(longest_run,
                episode_lengths.max())

        #
        # Update the status dictionary.
        #
        total_episodes = comm.allreduce(total_episodes, MPI.SUM)
        total_bs       = comm.allreduce(total_bs, MPI.SUM)

        for policy_id in self.policies:
            #
            # We didn't complete any episodes, so let's just take the top score
            # from our incomplete episode's scores.
            #
            if total_episodes < 1.0:
                top_rollout_score[policy_id] = max(top_rollout_score[policy_id],
                    ep_nat_scores[policy_id].max())

            top_score = top_rollout_score[policy_id]
            top_score = comm.allreduce(top_score, MPI.MAX)

            #
            # We used to keep track of the global reward range across all
            # episodes, but I think it's a bit more helpful to see the
            # fluctuations across rollouts.
            #
            max_nat_reward = rollout_max_nat_reward[policy_id]
            min_nat_reward = rollout_min_nat_reward[policy_id]
            max_nat_reward = comm.allreduce(max_nat_reward, MPI.MAX)
            min_nat_reward = comm.allreduce(min_nat_reward, MPI.MIN)

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

            num_agents = len(self.policies[policy_id].agent_ids)

            nat_reward_sum  = total_nat_scores[policy_id].sum()
            nat_reward_sum  = comm.allreduce(nat_reward_sum, MPI.SUM)

            total_scores_sum  = total_scores[policy_id].sum()
            total_scores_sum  = comm.allreduce(total_scores_sum, MPI.SUM)

            running_nat_reward = nat_reward_sum / total_episodes
            running_reward     = total_scores_sum / total_episodes
            rw_range           = (min_reward, max_reward)
            nat_rw_range       = (min_nat_reward, max_nat_reward)
            obs_range          = (min_obs, max_obs)

            global_top_reward = max(self.status_dict[policy_id]["top natural reward"],
                top_reward[policy_id])
            global_top_reward = comm.allreduce(global_top_reward, MPI.MAX)

            self.status_dict[policy_id]["score avg"]            = running_reward
            self.status_dict[policy_id]["natural score avg"]    = running_nat_reward
            self.status_dict[policy_id]["top score"]            = top_score
            self.status_dict[policy_id]["obs range"]            = obs_range
            self.status_dict[policy_id]["reward range"]         = rw_range
            self.status_dict[policy_id]["natural reward range"] = nat_rw_range
            self.status_dict[policy_id]["top natural reward"]   = global_top_reward
            self.status_dict[policy_id]["frozen"]               = self.policies[policy_id].frozen

            #
            # Update our bootstrap status info.
            #
            bs_rank_max = comm.allreduce(bs_max[policy_id], MPI.MAX)
            bs_rank_min = comm.allreduce(bs_min[policy_id], MPI.MIN)

            if total_bs == 0:
                bs_rank_avg = 0.0
            else:
                bs_rank_avg = comm.allreduce(bs_sum[policy_id].sum(), MPI.SUM) / total_bs

            self.status_dict[policy_id]["bootstrap range"] = (bs_rank_min, bs_rank_max)
            self.status_dict[policy_id]["bootstrap avg"]   = bs_rank_avg

            if self.policies[policy_id].enable_icm:
                intr_reward = total_intr_scores[policy_id].sum()
                intr_reward = comm.allreduce(intr_reward, MPI.SUM)

                ism = intr_reward / (total_episodes/ env_batch_size)
                self.status_dict[policy_id]["intrinsic score avg"] = ism.item()

                max_intr_reward = rollout_max_intr_reward[policy_id]
                min_intr_reward = rollout_min_intr_reward[policy_id]

                max_intr_reward = comm.allreduce(max_intr_reward, MPI.MAX)
                min_intr_reward = comm.allreduce(min_intr_reward, MPI.MIN)
                reward_range = (min_intr_reward, max_intr_reward)

                self.status_dict[policy_id]["intr reward range"] = reward_range

        longest_run      = comm.allreduce(longest_run, MPI.MAX)
        shortest_run     = comm.allreduce(shortest_run, MPI.MIN)
        total_rollout_ts = comm.allreduce(total_rollout_ts, MPI.SUM)
        avg_run          = comm.allreduce(avg_run, MPI.SUM) / num_procs

        self.status_dict["global status"]["total episodes"]  += total_episodes
        self.status_dict["global status"]["longest episode"]  = longest_run
        self.status_dict["global status"]["shortest episode"] = shortest_run
        self.status_dict["global status"]["average episode"]  = avg_run
        self.status_dict["global status"]["timesteps"]       += total_rollout_ts

        #
        # Finalize our datasets.
        #
        for policy_id in self.policies:
            self.policies[policy_id].finalize_dataset()

        comm.barrier()
        stop_time = time.time()
        self.status_dict["global status"]["rollout time"] = stop_time - start_time

    def learn(self, num_timesteps):
        """
        Learn!
            1. Create a rollout dataset.
            2. Update our networks.
            3. Repeat until we've reached our max timesteps.

        Parameters:
        -----------
        num_timesteps: int
            The maximum number of timesteps to run.
            Note that this is in addtion to however
            many timesteps were run during the last save.
        """
        start_time = time.time()
        ts_max     = self.status_dict["global status"]["timesteps"] + num_timesteps
        iteration  = self.status_dict["global status"]["iteration"]

        best_policy_scores = {}
        for policy_id in self.policies:
            best_policy_scores[policy_id] = -np.inf

        iter_start_time = time.time()
        iter_stop_time  = iter_start_time

        while self.status_dict["global status"]["timesteps"] < ts_max:

            self.freeze_scheduler()

            pre_rollout_timesteps = self.status_dict["global status"]["timesteps"]
            self.rollout()

            for policy_id in self.policies:
                current_score = self.status_dict[policy_id]["natural score avg"]

                if current_score >= best_policy_scores[policy_id]:
                    best_policy_scores[policy_id] = current_score
                    self.save(tag=f"{policy_id}_best")

            running_time    = (iter_stop_time - iter_start_time)
            running_time   += self.status_dict["global status"]["rollout time"]
            iter_start_time = time.time()

            self.status_dict["global status"]["running time"] += running_time

            self.print_status()
            self.save()

            if (iteration % self.checkpoint_every) == 0:
                self.save(tag=str(iteration))

            if self.save_train_scores:
                self._save_natural_score_avg(pre_rollout_timesteps)

            if self.save_ep_scores:
                self._save_full_len_episode_scores(pre_rollout_timesteps)

            if self.save_avg_ep_len:
                self._save_average_episode(pre_rollout_timesteps)

            if self.save_running_time:
                self._save_running_time(pre_rollout_timesteps)

            if self.save_bs_info:
                self._save_bs_info(pre_rollout_timesteps)

            data_loaders = {}
            for policy_id in self.policies:
                if not self.policies[policy_id].frozen:
                    data_loaders[policy_id] = DataLoader(
                        self.policies[policy_id].dataset,
                        batch_size = self.batch_size,
                        shuffle    = True)
                else:
                    data_loaders[policy_id] = None

            train_start_time = time.time()

            for policy_id in self.policies:
                self.policies[policy_id].train()

            #
            # We train each policy separately.
            #
            for policy_id in data_loaders:

                if self.policies[policy_id].frozen:
                    continue

                for epoch_idx in range(self.epochs_per_iter):
                    #
                    # arXiv:2006.05990v1 suggests that re-computing the
                    # advantages before each new epoch helps mitigate issues
                    # that can arrise from "stale" advantages.
                    #
                    if epoch_idx > 0 and self.recalc_advantages:
                        data_loaders[policy_id].dataset.recalculate_advantages()

                    self._ppo_batch_train(data_loaders[policy_id], policy_id)

                    if self.policies[policy_id].enable_icm:
                        self._icm_batch_train(data_loaders[policy_id], policy_id)

                    #
                    # Early ending using KL.
                    # NOTE: OpenAI's implementation multiplies the kl target
                    # by a magic number (1.5). I got sick of magic numbers and
                    # scrapped that approach.
                    #
                    comm.barrier()
                    if (self.status_dict[policy_id]["kl avg"] >
                        (self.policies[policy_id].target_kl)):

                        if self.verbose:
                            kl = self.policies[policy_id].target_kl
                            msg  = "Target KL of {} ".format(kl)
                            msg += "has been reached. "
                            msg += "Ending early (after "
                            msg += "{} epochs)".format(epoch_idx + 1)
                            rank_print(msg)
                        break

            #
            # We don't want to hange on to this memory as we loop back around.
            #
            for policy_id in self.policies:
                self.policies[policy_id].clear_dataset()

            #
            # Only use manual garbage collection if it's been requested.
            #
            if self.force_gc:
                del data_loaders
                gc.collect()

            now_time      = time.time()
            training_time = (now_time - train_start_time)
            self.status_dict["global status"]["train time"] = now_time - train_start_time

            iteration += 1
            self.status_dict["global status"]["iteration"] = iteration

            self.update_learning_rate()
            self.update_entropy_weight()

            comm.barrier()

            lr_sum = 0.0
            for policy_id in self.policies:
                lr_sum += self.policies[policy_id].lr()

            if lr_sum <= 0.0:
                rank_print("Learning rate has bottomed out. Terminating early")
                break

            iter_stop_time = time.time()

        stop_time   = time.time()
        seconds     = (stop_time - start_time)
        pretty_time = format_seconds(seconds)
        rank_print("Time spent training: {}".format(pretty_time))

    def _ppo_batch_train(self, data_loader, policy_id):
        """
        Train our PPO networks using mini batches.

        Parameters:
        -----------
        data_loader: PyTorch DataLoader
            A PyTorch data loader for a specific policy.
        policy_id: str
            The id for the policy that we're training.
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
                orig_shape = rewards_tg.shape
                rewards_tg = \
                    self.value_normalizers[policy_id].normalize(
                        rewards_tg.flatten()).reshape(orig_shape)

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

            data_loader.dataset.values[batch_idxs] = values.squeeze(-1).detach()

            curr_log_probs = curr_log_probs.flatten()
            log_probs      = log_probs.flatten()
            advantages     = advantages.flatten()
            entropy        = entropy.flatten()
            values         = values.flatten()
            rewards_tg     = rewards_tg.flatten()

            #
            # The heart of PPO: arXiv:1707.06347v2
            #
            ratios = torch.exp(curr_log_probs - log_probs)
            surr1  = ratios * advantages
            surr2  = torch.clamp(
                ratios, 1 - self.policies[policy_id].surr_clip,
                    1 + self.policies[policy_id].surr_clip) * advantages

            current_kl = (log_probs - curr_log_probs).mean().item()
            total_kl  += current_kl

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

                std = nn.functional.softplus(self.policies[policy_id].actor.distribution.log_std)
                rank_print("actor std: {}".format(std))

                comm.Abort()

            #
            # Calculate the actor loss.
            #
            actor_loss        = (-torch.min(surr1, surr2)).mean()
            total_actor_loss += actor_loss.item()

            if self.policies[policy_id].entropy_weight() != 0.0:
                total_entropy += entropy.mean().item()
                actor_loss -= \
                    self.policies[policy_id].entropy_weight() * entropy.mean()

            #
            # Optionally add a kl divergence penalty.
            #
            if self.policies[policy_id].kl_loss_weight > 0.0:
                actor_loss += self.policies[policy_id].kl_loss_weight * \
                    current_kl

            if values.size() == torch.Size([]):
                values = values.unsqueeze(0)
            else:
                values = values.squeeze()

            #
            # Calculate the critic loss. Optionally, we can use the clipped
            # version.
            #
            if self.policies[policy_id].use_huber_loss:
                critic_loss = nn.HuberLoss(delta=10.0)(values, rewards_tg)
            else:
                critic_loss = nn.MSELoss()(values, rewards_tg)

            #
            # This clipping strategy comes from arXiv:2005.12729v1, which
            # differs somewhat from other implementations (rllib for example)
            # but should be true to OpenAI's original approach.
            #
            if self.policies[policy_id].vf_clip is not None:
                clipped_values = torch.clamp(
                    values,
                    -self.policies[policy_id].vf_clip,
                    self.policies[policy_id].vf_clip)

                if self.user_huber_loss:
                    clipped_loss = nn.HuberLoss(delta=10.0)(clipped_values, rewards_tg)
                else:
                    clipped_loss = nn.MSELoss()(clipped_values, rewards_tg)
                critic_loss  = torch.max(critic_loss, clipped_loss)

            total_critic_loss += critic_loss.item()

            #
            # Let the policies update their weights given the actor
            # and critic losses.
            #
            self.policies[policy_id].update_weights(actor_loss, critic_loss)

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

                data_loader.dataset.actor_hidden[batch_idxs]  = actor_hidden
                data_loader.dataset.critic_hidden[batch_idxs] = critic_hidden

                data_loader.dataset.actor_cell[batch_idxs]  = actor_cell
                data_loader.dataset.critic_cell[batch_idxs] = critic_cell

            comm.barrier()
            counter += 1

        counter           = comm.allreduce(counter, MPI.SUM)
        total_entropy     = comm.allreduce(total_entropy, MPI.SUM)
        total_actor_loss  = comm.allreduce(total_actor_loss, MPI.SUM)
        total_critic_loss = comm.allreduce(total_critic_loss, MPI.SUM)
        total_kl          = comm.allreduce(total_kl, MPI.SUM)
        w_entropy = total_entropy * self.policies[policy_id].entropy_weight()

        self.status_dict[policy_id]["weighted entropy"] = w_entropy / counter
        self.status_dict[policy_id]["actor loss"] = \
            total_actor_loss / counter

        self.status_dict[policy_id]["critic loss"] = \
            total_critic_loss / counter

        self.status_dict[policy_id]["kl avg"] = total_kl / counter

    def _icm_batch_train(self, data_loader, policy_id):
        """
        Train our ICM networks using mini batches.

        Parameters:
        ----------
        data_loader: PyTorch DataLoader
            A PyTorch data loader for a specific policy.
        policy_id: str
            The id for the policy that we're training.
        """
        total_icm_loss = 0
        counter = 0

        for batch_data in data_loader:

            _, obs, next_obs, _, actions, _, _, _, _, _, _, _, _ =\
                batch_data

            torch.cuda.empty_cache()

            #
            # We have some cases to consider:
            #  1. Normal case. Data comes in with shape (batch_size, *), where
            #     batch size includes agents.
            #  2. We're using agent shared ICM. In this case, we need our
            #     data to have shape (batch_size, num_agents * data_size), BUT
            #     the agent's need to be in a specific order.
            #  3. We're using standard ICM, but our policy has agent_grouping
            #     enabled. In this case, our data comes in with shape
            #     (batch_size, num_agents, *), and we need to reshape it
            #     into (batch_size * num_agents, *).
            #
            if self.policies[policy_id].agent_shared_icm:

                obs      = obs[:, self.policies[policy_id].agent_idxs, :]
                next_obs = next_obs[:, self.policies[policy_id].agent_idxs, :]

                #
                # Discrete actions will often arrive with shape
                # (batch_size, num_agents).
                #
                if len(actions.shape) < 3:
                    a_shape = actions.shape
                    actions = actions.reshape((a_shape[0], a_shape[1], 1))

                actions  = actions[:, self.policies[policy_id].agent_idxs, :]

                batch_size = obs.shape[0]
                obs        = obs.reshape((batch_size, -1))
                next_obs   = next_obs.reshape((batch_size, -1))
                actions    = actions.reshape((batch_size, -1))

            elif self.policies[policy_id].agent_grouping:
                batch_size = obs.shape[0] * obs.shape[1]
                obs        = obs.reshape((batch_size, -1))
                next_obs   = next_obs.reshape((batch_size, -1))
                actions    = actions.reshape((batch_size, -1))

            if len(actions.shape) < 2:
                actions = actions.unsqueeze(1)

            _, inv_loss, f_loss = self.policies[policy_id].icm_model(
                obs, next_obs, actions)

            icm_loss = (((1.0 - self.policies[policy_id].icm_beta) * f_loss) +
                (self.policies[policy_id].icm_beta * inv_loss))

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

    def save(self, tag="latest"):
        """
        Save all information required for a restart.

        Parameters:
        -----------
        tag: str
            A tag representing when this save took place.
        """
        if self.verbose:
            rank_print("Saving state")

        if self.test_mode:
            msg = "WARNING: save() was called while in test mode. Disregarding."
            rank_print(msg)
            return

        if rank == 0:
            #
            # NOTE: these save methods will save and load from whatever
            # ranks they're called from, and they will go into <name>_<rank>.ext
            # files. We only want to save on rank 0 in this case, only call
            # from rank 0.
            #
            os.makedirs(os.path.join(self.env_info_path, tag), exist_ok=True)

            if self.save_env_info and self.env != None:
                self.env.save_info(os.path.join(self.env_info_path, tag))

            for policy_id in self.policies:
                self.policies[policy_id].save(self.state_path, tag)

            if self.normalize_values:
                for policy_id in self.value_normalizers:
                    self.value_normalizers[policy_id].save_info(
                        os.path.join(self.env_info_path, tag))

            self.freeze_scheduler.save_info()

            #
            # TODO: we might want to save the status info using
            # tags as well.
            #
            file_name  = "state_0.pickle"
            state_file = os.path.join(self.state_path, file_name)
            with open(state_file, "wb") as out_f:
                pickle.dump(self.status_dict, out_f,
                    protocol=pickle.HIGHEST_PROTOCOL)

        comm.barrier()

    def load_status(self, state_path):
        """
        Load our status dictionary from a restart.

        Parameters:
        -----------
        state_path: str
            The path to our saved state.

        Returns:
        --------
        dict:
            The loaded status dictionary.
        """
        file_name  = "state_0.pickle"
        state_file = os.path.join(state_path, file_name)

        with open(state_file, "rb") as in_f:
            tmp_status_dict = pickle.load(in_f)

        return tmp_status_dict

    def load_policy(self, policy_id, state_path, tag="latest"):
        """
        Load our policies and related state from a checkpoint.

        Parameters:
        -----------
        policy_id: str
            The id of the policy to load.
        state_path: str
            The state path to load the policy from.
        tag: str
            A tag representing when this save took place.
        """
        self.policies[policy_id].load(state_path, tag)

        if self.normalize_values and policy_id in self.value_normalizers:
            try:
                self.value_normalizers[policy_id].load_info(
                            os.path.join(self.env_info_path, tag))
            except:
                pass

    def direct_load_policy(self, policy_id, policy_path):
        """
        Load our policies and related state from a checkpoint.

        Parameters:
        -----------
        policy_id: str
            The id of the policy to load.
        state_path: str
            The state path to load the policy from.
        """
        self.policies[policy_id].direct_load(policy_path)

        state_path = Path(policy_path).parent.parent.absolute()
        tag        = Path(policy_path).parent.name

        try:
            if self.normalize_values and policy_id in self.value_normalizers:
                self.value_normalizers[policy_id].load_info(
                    os.path.join(self.env_info_path, tag))
        except:
            if self.normalize_values and policy_id in self.value_normalizers:
                self.value_normalizers[policy_id].load_info(
                    os.path.join(self.env_info_path, "latest"))

    def load_policies(self, state_path, tag):
        """
        Load our policies and related state from a checkpoint.

        Parameters:
        -----------
        state_path: str
            The state path to load the policies from.
        tag: str
            A tag representing when this save took place.
        """
        for policy_id in self.policies:
            self.load_policy(policy_id, state_path, tag)

    def load_env_info(self, env_info_path, tag="latest"):
        """
        Load our environment info.

        Parameters:
        -----------
        env_info_path: str
            The path to where the env info was saved.
        tag: str
            A tag representing when this save took place.
        """
        if self.save_env_info and self.env != None:
            self.env.load_info(os.path.join(env_info_path, tag))
        
    def load(self, state_path, tag):
        """
        Load all information required for a restart.

        Parameters:
        -----------
        state_path: str
            The state path to load from.
        tag: str
            A tag representing when this save took place.
        """
        self.load_policies(state_path, tag)
        self.load_env_info(state_path, tag)
        return self.load_status(state_path)

    def _write_timestep_scores_to_file(self, filename, timestep, score):
        """
        Write a single score and associated timestep to a numpy file
        for future analysis.

        Parameters:
        -----------
        filename: str
            The name of the file to save curve data to.
        timestep: int/float
            The timestep to associate with this score.
        score: float
            The score to save.
        """
        with open(filename, "ab") as out_f:
            out_data    = np.zeros((1, 2))
            out_data[0][0] = timestep
            out_data[0][1] = score
            np.savetxt(out_f, np.array(out_data))

    def _save_full_len_episode_scores(self, timestep):
        """
        Save the "full length episode" scores. These scores will not be
        truncated during the rollouts. Instead, they are collected across
        rollouts, once the episode has terminated.

        Parameters:
        -----------
        timestep: int
            The timestep to correlate with the current scores.
        """
        episode_scores = self.full_len_ep_scores.get_mean_scores()

        if rank == 0:
            for policy_id in episode_scores:
                score = episode_scores[policy_id]
                score_f = os.path.join(self.ep_score_path,
                    f"{policy_id}_scores.npy")

                self._write_timestep_scores_to_file(score_f, timestep, score)

    def _save_natural_score_avg(self, timestep):
        """
        Save the natural reward averages of each policy to numpy
        files.

        Parameters:
        -----------
        timestep: int
            The timestep to correlate with the current scores.
        """
        if rank == 0:
            for policy_id in self.policies:
                score = self.status_dict[policy_id]["natural score avg"]
                score_f = os.path.join(self.train_score_path,
                    f"{policy_id}_scores.npy")

                self._write_timestep_scores_to_file(score_f, timestep, score)

    def _save_average_episode(self, timestep):
        """
        Save the average episode length to a numpy txt file.

        Parameters:
        -----------
        timestep: int
            The timestep to correlate with the current scores.
        """
        if rank == 0:
            avg_ep   = self.status_dict["global status"]["average episode"]
            len_file = os.path.join(self.ep_len_path,
                f"average_episode.npy")

            self._write_timestep_scores_to_file(len_file, timestep, avg_ep)

    def _save_running_time(self, timestep):
        """
        Save the current running time to a numpy txt file.

        Parameters:
        -----------
        timestep: int
            The timestep to correlate with the current scores.
        """
        if rank == 0:
            runtime      = self.status_dict["global status"]["running time"]
            runtime_file = os.path.join(self.runtime_path,
                f"running_time.npy")

            self._write_timestep_scores_to_file(runtime_file, timestep, runtime)

    def _save_bs_info(self, timestep):
        """
        Save the current bootstrap min, max, and avg.

        Parameters:
        -----------
        timestep: int
            The timestep to correlate with the current scores.
        """
        if rank == 0:

            for policy_id in self.policies:
                bs_min  = self.status_dict[policy_id]["bootstrap range"][0]
                bs_file = os.path.join(self.bs_min_path,
                    f"{policy_id}_bs_min.npy")

                self._write_timestep_scores_to_file(bs_file, timestep, bs_min)

                bs_max  = self.status_dict[policy_id]["bootstrap range"][1]
                bs_file = os.path.join(self.bs_max_path,
                    f"{policy_id}_bs_max.npy")

                self._write_timestep_scores_to_file(bs_file, timestep, bs_max)

                bs_avg  = self.status_dict[policy_id]["bootstrap avg"]
                bs_file = os.path.join(self.bs_avg_path,
                    f"{policy_id}_bs_avg.npy")

                self._write_timestep_scores_to_file(bs_file, timestep, bs_avg)

    def set_test_mode(self, test_mode):
        """
        Enable or disable test mode in all required modules.

        Parameters:
        -----------
        test_mode: bool
            A bool representing whether or not to enable test mode.
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
        --------
        dict:
            The state dictionary minus the environment.
        """
        state = self.__dict__.copy()
        del state["env"]
        del state["test_mode_dependencies"]
        return state

    def __setstate__(self, state):
        """
        Override the setstate method for pickling.

        Parameters:
        -----------
        state: dict
            The state loaded from a pickled PPO object.
        """
        self.__dict__.update(state)
        self.env = None
        self.test_mode_dependencies = self.pickle_safe_test_mode_dependencies

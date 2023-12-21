# PPO And Friends

PPO and Friends is a PyTorch implementation of Proximal Policy Optimation
along with various extra optimizations and add-ons (freinds).

We are currently compatible with the following environment frameworks:
* Gymnasium
* Gym (including versions <= 0.21)
* PettingZoo
* Abmarl Gridworld

# Our Friends

Some of our friends:

* Decentralized Distributed Proximal Policy Optimization (DD-PPO)
* Intrinsic Curiosity Module (ICM)
* Multi Agent Proximal Policy Optimization (MAPPO)
* Multi-Agent Transformer (MAT)
* Generalized Advantage Estimations (GAE)
* LSTM
* Gradient, reward, bootstrap, value, and observation clipping
* KL based early ending
* KL punishment
* Observation, advantage, and reward normalization
* Advantage re-calculation
* Vectorized environments

For a full list of policy options and their defaults, see 
`ppo_and_friends/policies/`

Note that this implementation of PPO uses separate networks for critics
and actors (except for the Multi-Agent Transformer).

# Installation

Simpy issue the following command from the
top directory to install the standard PPO-And-Friends library.
```
pip install .
```

Optionally, you can install the following extensions as well. Note that these extensions
are not always cross-compatible. For instance, `abmarl` support is not currently compatible
with `gym_baselines` support.

```
# Install support for Abmarl
pip install .[abmarl]

# Install support running the (old) gym environments in baselines/gym
pip install --upgrade pip wheel==0.38.4
pip install .[gym]

# Install support for running the gymnasium environments in baselines/gymnasium
pip install .[gymnasium]
```

# Terminology
Terminology varies across implemenations and publications, so here are
some commonly overloaded terms and how we define them.

1. **batch size**: we refer to the gradient descent mini-batch size as the
   batch size. This is sometimes referred to as 'mini batch size',
   'sgd mini batch size', etc. This is defined as `batch_size` in our code.
2. **timesteps per rollout**: this refers to the total number of timesteps
   collected in a single rollout. This is sometimes referred to as the batch
   size. This includes the data collected from all processors.
   The exception to this rule is that a single processor will only see the
   number timesteps it needs to collect. This is defined as `ts_per_rollout`
   in our code.
3. **max timesteps per episode**: this refers to the maximum number of
   timesteps collected for a single episode trajectory. This is sometimes
   referred to as horizon or trajectory length. If max timesteps
   per episode is 10, and we're collecting 100 timesteps in our rollout
   on a single processor, then we'll end up with 10 episodes of length 10.
   Note that the environment does not need to enter a done state for
   an episode's trajectory to end. This is defined as `max_ts_per_ep` in our
   code.

# Policy Options

This implementation of PPO supports both single and multi-agent environments, and,
as such, there are many design decisions to made. Currently, ppo-and-friends
follows the standards outlined below.

1. All actions sent to the step function will be wrapped in a dictionary
   mapping agent ids to actions.
2. Calling `env.step(actions)` will result in a tuple of the following form:
   `(obs, critic_obs, reward, info, done)` s.t. each tuple element is a
   dictionary mapping agent ids to the appropriate data.
3. Death masking is used at all times, which means that all agents are
   expected to exist in the `step` results as long as an episode hasn't
   terminated.

Since not all environments will adhere to the above standards, various
wrappers are provided in the `environments/` directory. For best results,
all environments should be wrapped in a class inherting from
`PPOEnvironmentWrapper`.

## PPO
This is the default policy for single-agent environments.

## MAPPO and IPPO
arXiv:2103.01955v4 makes the distinction between MAPPO and IPPO such that
the former uses a centralized critic receiving global information about
the agents of a shared policy (usually a concatenation of the observations),
and the later uses an independent, decentralized critic.

Both options can be enabled by setting the `critic_view` parameter in
the `PPOEnvironmentWrapper` appropriately. Options as of now are
"global", "policy", and "local".

* global: this option will send observations from ALL agents in the environment,
  regardless of which policy they belong to, to every critic. Note that, when using
  a single policy, this is identical to MAPPO. However, when using multiple policies,
  each critic can see the observations of other policies.
* policy: this option will combine observations from all agents under shared policies,
  and the critics of those policies will receive the shared observations. This option
  is identical to MAPPO when using a single policy, and it alows for similar behavior
  when using multiple polices (multiple policies was not convered in the paper, but
  this general concept translates well).
* local: this option will send local observations from each agent to the critic of
  their respective policy. This is IPPO when using a single policy with multiple agents
  and PPO when using a single policy with one agent.

All multi-agent environment wrappers that inherit from `PPOEnvironmentWrapper`
allow users to set `critic_view` with the exception of `MAT`, which cannot
decouple the critic's from the actors' observations.

## Multi-Agent Transformer

The Multi-Agent Transformer (MAT) can be enabled by setting a policie's class
to MATPolicy. Different policy classses can be used for different policies
within the same game. For instance, you can have one team use MATPolicy
and another team use PPOPolicy.

The implemenation of MAT within PPO-AF follows the original publication as
closely as possible. Some exceptions were made to account for differences
between the publication and it's associated source code and differences
in architecture between PPO-AF and the publication's source code.

Full details on MAT can be found at its official site:
https://sites.google.com/view/multi-agent-transformer

# Environment Wrappers

## Gymnasium

Both single agent and multi-agent gymnasium games are supported through
the `SingleAgentGymWrapper` and `MultiAgentGymWrapper`, respectively.
For examples on how to train a gymnasium environment, check out the runners
in `baselines/gymnasium/`.

**IMPORTANT**: While Gymnasium does not have a standard interface for multi-agent games,
I've found some commonalities among many publications, and we are using this
as our standard. You may need to make changes to your multi-agent gymnasium
environments before they can be wrapped in the `MultiAgentGymWrapper`.

Our expectaions of multi-agent Gymnasium environments are as follows:
* The step method must return observation, reward, terminated, truncated, info.
  observation, reward, terminated, and truncated
  must be iterables s.t. each index maps to a specific agent, and this order
  must not change. info must be a dict.
* The reset method must return the agent observations as an iterable with
  the same index constraints defined above.
* Both `env.observation_space` and `env.action_space` must be iterables
  such that indices map to agents in the same order they are given from
  the step and reset methods.

## Gym <= 0.21

For environments that only exist in versions <= 0.21 of Gym, you
can use the `Gym21ToGymnasium` wrapper. See `baselines/gym/`
for examples.

**IMPORTANT**: While Gym does not have a standard interface for multi-agent games,
I've found some commonalities among many publications, and we are using this
as our standard. You may need to make changes to your multi-agent gymnasium
environments before they can be wrapped in the `MultiAgentGymWrapper`.

Our expectaions of multi-agent Gym environments are as follows:
* The step method must return observation, reward, done, info.
  observation, reward, and done
  must be iterables s.t. each index maps to a specific agent, and this order
  must not change. info must be a dict.
* The reset method must return the agent observations as an iterable with
  the same index constraints defined above.
* Both `env.observation_space` and `env.action_space` must be iterables
  such that indices map to agents in the same order they are given from
  the step and reset methods.

## Gym To Gymnasium

Games that exist in Gym versions >= 0.26 but not Gymnasium can be tricky.
I've found that the biggest issue is the spaces not matching up. We have
a function `gym_space_to_gymnasium_space` in `environments/gym/version_wrappers.py`
that can be used to (attempt to) convert spaces from Gym to Gymnasium.

## Abmarl

The `AbmarlWrapper` can be used for Abmarl environments. See `baselines/abmarl` for
examples.

## Petting Zoo

The `ParallelZooWrapper` can be used for PettingZoo environments. See `baselines/pettingzoo`
for examples.

## Custom

All environments must be wrapped in the `PPOEnvironmentWrapper`. If you're
using a custom environment that doesn't conform to supported standards,
you can create your own wrapper that inherits from `PPOEnvironmentWrapper`,
found in `environments/ppo_env_wrappers.py`.

# MPI And Environments Per Processor
Both MPI and multiple environment instances per processor are supported,
and utilizing these options can greatly speed up training time. Some
environments may be sensitive to the choices here, which can have an
impact on training. See the **Tips and Tricks** section for some general
setting suggestions.

Currently, the default is to use GPUs when training on a single processor
and CPUs when training on multiple processors. This can be overridden with
the `--alow_mpi_gpu` flag, which is helpful for environments that require
networks that can benefit from GPUs (convolutions, for example).

NOTE: the current implementation of multiple environment instances per
processor assumes that the rollout bottleneck will come from inference rather
than stepping through the environment. Because of this, the multiple environment
instances are run in succession rather than in parallel, and the speed up
comes from batched inference during the rollout. Very slow environments may
not see a performance gain from increasing `envs_per_proc`.

**Usage:**

mpirun:
```
mpirun -n {num_procs} ppoaf --envs_per_proc {envs_per_proc} ...
```

srun:
```
srun -N1 -n {num_procs} ppoaf --envs_per_proc {envs_per_proc} ...
```

Some things to note:
1. The total timesteps per rollout is divided among the processors. So,
   if the environment is set to run 1024 timesteps per rollout, each
   processor will collect 1024/N of those timesteps, where N is the total
   number of processors (remainders go to processor 0). Note that there
   are various side effects of this, some of which are outlined below.
   Also, `envs_per_proc` can have a similar effect on reducing the total
   timesteps that each environment instance experiences, especially if
   each instance can reach its max timesteps before being "done".
2. Increasing the processor count doesn't always increase training speed.
   For instance, imagine an environment that can only reach unique states
   in the set `U` after running for at least 500 time steps. If our total
   timesteps per rollout is set to 1024, and we run with > 2 processors,
   we will never collect states from `U` and thus might not ever learn
   how to handle those unique situations. A similar logic applies for
   `envs_per_proc`. **Note**: this particular issue is now partially
   mitigated by the recent addition of "soft resets", but this feature
   has its own complications.
3. When running with multiple processors or environment instances,
   the stats that are displayed might not fully reflect the true status
   of learning. For instance, imagine an environment that, when performing
   well, receives +1 for every timestep and is allowed to run for a
   maximum of 100 timesteps. This results in a max score of +100. If each
   processor is only collecting 32 timesteps per rollout, the highest
   score any of them could ever achieve would be 32. Therefore, a reported
   score around 32 might actually signal a converged policy.


# Environment Runners

To train an environment, an `EnvironmentRunner` must first be defined. The
runner will be a class that inherits from
`ppo_and_friends.runners.env_runner.EnvironmentRunner` or the `GymRunner`
located within the same module. The only method you need to define is
`run`, which should call `self.run_ppo(...)`.


**Example**:
```
import gymnasium as gym
from ppo_and_friends.environments.gym.wrappers import SingleAgentGymWrapper
from ppo_and_friends.policies.utils import get_single_policy_defaults
from ppo_and_friends.runners.env_runner import GymRunner
from ppo_and_friends.networks.actor_critic_networks.feed_forward import FeedForwardNetwork
from ppo_and_friends.utils.schedulers import *
import torch.nn as nn
from ppo_and_friends.runners.runner_tags import ppoaf_runner

@ppoaf_runner
class CartPoleRunner(GymRunner):

    def add_cli_args(self, parser):
        """
        Define extra args that will be added to the ppoaf command.

        Parameters:
        -----------
        parser: argparse.ArgumentParser
            The parser from ppoaf.

        Returns:
        --------
        argparse.ArgumentParser:
            The same parser as the input with potentially new arguments added.
        """
        parser.add_argument("--learning_rate", type=float, default=0.002)
        return parser

    def run(self):

        env_generator = lambda : \
            SingleAgentGymWrapper(gym.make('CartPole-v0',
                render_mode = self.get_gym_render_mode()))

        actor_kw_args = {}
        actor_kw_args["activation"] = nn.LeakyReLU()
        critic_kw_args = actor_kw_args.copy()

        lr = 0.0002
        ts_per_rollout = self.get_adjusted_ts_per_rollout(256)

        policy_args = {\
            "ac_network"       : FeedForwardNetwork,
            "actor_kw_args"    : actor_kw_args,
            "critic_kw_args"   : critic_kw_args,
            "lr"               : self.cli_args.lr,
        }

        policy_settings, policy_mapping_fn = get_single_policy_defaults(
            env_generator = env_generator,
            policy_args   = policy_args)

        save_when = ChangeInStateScheduler(
            status_key     = "extrinsic score avg",
            status_preface = "single_agent",
            compare_fn     = np.greater_equal,
            persistent     = True)

        self.run_ppo(**self.kw_run_args,
                     save_when          = save_when,
                     env_generator      = env_generator,
                     policy_settings    = policy_settings,
                     policy_mapping_fn  = policy_mapping_fn,
                     batch_size         = 256,
                     ts_per_rollout     = ts_per_rollout,
                     max_ts_per_ep      = 32,
                     obs_clip           = (-10., 10.),
                     reward_clip        = (-10., 10.),
                     normalize_obs      = True,
                     normalize_rewards  = True,
                     normalize_adv      = True)
```

**Make note of the following requirements**:
1. your environment MUST be wrapped in one of the available ppo-and-friends
   environment wrappers. Currently available wrappers are SingleAgentGymWrapper,
   MultiAgentGymWrapper, AbmarlWrapper, and ParallelZooWrapper.
2. You must add the `@ppoaf_runner` decorator to your class.

See the `baselines` directory for more examples.

# Training And Testing

To train an environment, use the following command:
```
ppoaf train <path_to_runner_file>
```

Running the same command again will result in loading the previously
saved state. You can re-run from scratch by using the `--clobber` option.

A complete list of options can be seen with the `help` command:
```
ppoaf --help
```

To test a model that has been trained on a particular environment,
you can issue the following command:
```
ppoaf test <path_to_output_directory> --num_test_runs <num_test_runs> --render
```

By default, exploration is disabled during testing, but you can enable it
with the `--test_explore` flag. Example:

```
ppoaf test <path_to_output_directory> --num_test_runs <num_test_runs> --render --test_explore
```
The output directory will be given the same name as your runner file, and
it will appear in the path specified by `--state_path` when training, which
defaults to `./saved_states`.

Note that enabling exploration during testing will have varied results. I've found
that most of the environments I've tested perform better without exploration, but
there are some environments that will not perform at all without it.

# Plotting Results
If `--save_train_scores` is used while training, the results can be plotted using
PPO-And-Friend's ploting utility.

```
ppoaf plot path1 path2 path3 ... <options>
```

# Tips And Tricks

**Action Space**
By default, predictions in the continuous action space will be in the [-1, 1]
range rather than infering from the environment's action space. This is to
avoid issues with unbounded spaces. The range of the continuous action space
can easily be configured through the actor keyword arguments to the policy.

For example, the following will set the action space to the range [-100, 100]:
```
...
actor_kw_args["distribution_min"] = -100.
actor_kw_args["distribution_max"] = 100.

policy_args = {\
    "actor_kw_args"    : actor_kw_args,
}

policy_settings = { "actor_0" : \
    (None,
     env_generator().observation_space["actor_0"],
     env_generator().critic_observation_space["actor_0"],
     env_generator().action_space["actor_0"],
     policy_args),
}
...
```

See `baselines/humanoid.py` for a more concrete example.

**Performance**

Unless otherwise stated, all provided environments should be capable of
training "successful" policies. When it's provided, the definition of
success comes from the environment itself. BipedalWalker, for instance, defines
success as reaching an average score >= 300 over 100 test iterations. For
environments that do not provide definitions of success, I wing it.

Open-AI refers to environments that don't have a definition for "solved"
as unsolved environments. I think this is a bit missleading, as these
environments often have distinct goals that can be accomplished. For instance,
Pendulum is an unsolved environment, but we also know what the goal is, and
we can accomplish that goal very easily with RL. Currently, the only
environment in this repositroy that I haven't seen accomplish it's goal
is HumanoidStandup.

If any environment is performing poorly, I'd suggest trying a different seed.
If that doesn't work, feel free to open a ticket.
Of course, different systems will also result in different performance.
For comparison's sake, here is my system info:

OS:
```
$ Linux pop-os 5.15.5-76051505-generic
```
GPU:
```
GP104BM [GeForce GTX 1070 Mobile]
```

# Baselines

The `baselines` directory contains a number of pre-defined `EnvironmentRunners`
that can be used as references.

Policies can differ from one training to another, and the longer training
sessions generally result in better policies. For the results demonstrated
below, I trained for a moderate amount, which is usually just enough to
see a decent policy. See the **Environment Setttings** section for details.

## CartPole
<img src="https://github.com/aowen87/ppo_and_friends/blob/main/gifs/CartPole.gif" width="300" height="200" />

- **test score: 200**
- **average over 100 test runs: 200**

## Pendulum
<img src="https://github.com/aowen87/ppo_and_friends/blob/main/gifs/Pendulum.gif" width="300" height="200" />

- **test score: -241.6**

## Acrobot
<img src="https://github.com/aowen87/ppo_and_friends/blob/main/gifs/Acrobot.gif" width="300" height="200" />

- **test score: -82**

## MountainCar
<img src="https://github.com/aowen87/ppo_and_friends/blob/main/gifs/MountainCar.gif" width="300" height="200" />

- **test score: -108**
- **average over 100 test runs: -105.1**

## MountainCarContinuous
<img src="https://github.com/aowen87/ppo_and_friends/blob/main/gifs/MountainCarContinuous.gif" width="300" height="200" />

- **test score: 94.6**
- **average over 100 test runs: 92.0**

## LunarLander
<img src="https://github.com/aowen87/ppo_and_friends/blob/main/gifs/LunarLander.gif" width="300" height="200" />

- **test score: 259.4**

## LunarLanderContinuous
<img src="https://github.com/aowen87/ppo_and_friends/blob/main/gifs/LunarLanderContinuous.gif" width="300" height="200" />

- **test score: 281.7**

## BipedalWalker
<img src="https://github.com/aowen87/ppo_and_friends/blob/main/gifs/BipedalWalker.gif" width="300" height="200" />

- **test score: 326.2**
- **average over 100 test runs: ~319**

## BipedalWalkerHardcore
<img src="https://github.com/aowen87/ppo_and_friends/blob/main/gifs/BipedalWalkerHardcore.gif" width="300" height="200" />

- **test score: 329.7**
- **average over 100 test runs: ~313**

## BreakoutRAM
<img src="https://github.com/aowen87/ppo_and_friends/blob/main/gifs/BreakoutRAM.gif" width="300" height="200" />

- **test score: N/A**: I cut off the test at 500 steps for this gif.

## InvertedPendulum
<img src="https://github.com/aowen87/ppo_and_friends/blob/main/gifs/InvertedPendulum.gif" width="300" height="200" />

- **test score: 1000**

## InvertedDoublePendulum
<img src="https://github.com/aowen87/ppo_and_friends/blob/main/gifs/InvertedDoublePendulum.gif" width="300" height="200" />

- **test score: 9318.5**

## Ant
<img src="https://github.com/aowen87/ppo_and_friends/blob/main/gifs/Ant.gif" width="300" height="200" />

- **test score: 6106.2**
- **average over 100 test runs: 6298.3**

## Walker2d
<img src="https://github.com/aowen87/ppo_and_friends/blob/main/gifs/Walker2d.gif" width="300" height="200" />

- **test score: 3530.0**

## Hopper
<img src="https://github.com/aowen87/ppo_and_friends/blob/main/gifs/Hopper.gif" width="300" height="200" />

- **test score: 3211.0**

## Swimmer
<img src="https://github.com/aowen87/ppo_and_friends/blob/main/gifs/Swimmer.gif" width="300" height="200" />

- **test score: 131.3**

## HalfCheetah
<img src="https://github.com/aowen87/ppo_and_friends/blob/main/gifs/HalfCheetah.gif" width="300" height="200" />

- **test score: 4157.9**

## Humanoid
<img src="https://github.com/aowen87/ppo_and_friends/blob/main/gifs/Humanoid.gif" width="300" height="200" />

- **test score: 6330.9**

## RobotWarehouseTiny
<img src="https://github.com/aowen87/ppo_and_friends/blob/main/gifs/RobotWarehouseTiny.gif" width="300" height="200" />

- **test score (averaged across all agents): 11.0**

## LevelBasedForaging
<img src="https://github.com/aowen87/ppo_and_friends/blob/main/gifs/LevelBasedForaging.gif" width="300" height="200" />

- **test score (averaged across all agents): 0.25**
- **highest test score (max across all agents): 0.33**

## PressurePlate
<img src="https://github.com/aowen87/ppo_and_friends/blob/main/gifs/PressurePlate.gif" width="300" height="200" />

- **test score (averaged across all agents): -19.27**

## MPESimpleAdversary
<img src="https://github.com/aowen87/ppo_and_friends/blob/main/gifs/MPESimpleAdversary.gif" width="300" height="200" />

- **test scores for individual agents:**

```
Agent adversary_0:
    Ran env 3 times.
    Ran 96 total time steps.
    Ran 32.0 time steps on average.
    Lowest score: -17.08993209169898
    Highest score: -7.286026444287251
    Average score: -11.464761470637379

Agent agent_0:
    Ran env 3 times.
    Ran 96 total time steps.
    Ran 32.0 time steps on average.
    Lowest score: 1.3052540572620759
    Highest score: 8.179996521593692
    Average score: 3.964994910365373

Agent agent_1:
    Ran env 3 times.
    Ran 96 total time steps.
    Ran 32.0 time steps on average.
    Lowest score: 1.3052540572620759
    Highest score: 8.179996521593692
    Average score: 3.964994910365373
```

## MPESimpleTag
<img src="https://github.com/aowen87/ppo_and_friends/blob/main/gifs/MPESimpleTag.gif" width="300" height="200" />

- **test scores for individual agents:**
```
Agent adversary_0:
    Ran env 3 times.
    Ran 384 total time steps.
    Ran 128.0 time steps on average.
    Lowest score: 180.0
    Highest score: 280.0
    Average score: 226.66666666666666

Agent adversary_1:
    Ran env 3 times.
    Ran 384 total time steps.
    Ran 128.0 time steps on average.
    Lowest score: 180.0
    Highest score: 280.0
    Average score: 226.66666666666666

Agent adversary_2:
    Ran env 3 times.
    Ran 384 total time steps.
    Ran 128.0 time steps on average.
    Lowest score: 180.0
    Highest score: 280.0
    Average score: 226.66666666666666

Agent agent_0:
    Ran env 3 times.
    Ran 384 total time steps.
    Ran 128.0 time steps on average.
    Lowest score: -309.1990745961666
    Highest score: -211.49080680310726
    Average score: -252.04373224576315
```

## MPESimpleSpread
<img src="https://github.com/aowen87/ppo_and_friends/blob/main/gifs/MPESimpleSpread.gif" width="300" height="200" />

- **test scores for individual agents:**
```
Agent agent_0:
    Ran env 5 times.
    Ran 320 total time steps.
    Ran 64.0 time steps on average.
    Lowest score: -20.294600811907934
    Highest score: -10.20188162020333
    Average score: -17.444208170487293

Agent agent_1:
    Ran env 5 times.
    Ran 320 total time steps.
    Ran 64.0 time steps on average.
    Lowest score: -20.294600811907934
    Highest score: -10.20188162020333
    Average score: -16.74420817048731

Agent agent_2:
    Ran env 5 times.
    Ran 320 total time steps.
    Ran 64.0 time steps on average.
    Lowest score: -20.294600811907934
    Highest score: -10.20188162020333
    Average score: -17.344208170487292
```

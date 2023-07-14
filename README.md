# PPO And Friends

PPO and Friends is a PyTorch implementation of Proximal Policy Optimation
along with various extra optimizations and add-ons (freinds).

While this project is intended to be compatible with Gymnasium environments
and interfaces, you'll occasionally see situtations where utilities that gym
(or stable baselines) provides have been ignored in favor of creating our
own versions of these utilities. These choices are made to more easily handle
environments and/or algorithms that don't follow the standard rules and
assumptions enforced by existing frameworks.

# Our Friends

Some of our friends:

* Intrinsic Curiosity Module (ICM)
* Multi Agent Proximal Policy Optimization (MAPPO)
* Generalized Advantage Estimations (GAE)
* LSTM integration into PPO algorithm
* Gradient, reward, bootstrap, value, and observation clipping
* KL based early ending
* KL punishment
* Splitting observations by proprioceptive and exteroceptive information
* Observation, advantage, and reward normalization
* Learning rate annealing
* Entropy annealing
* Intrinsic reward weight annealing
* Vectorized environments
* Observational augmentations

For a full list of policy options and their defaults, see 
`ppo_and_friends/policies/agent_policy.py`.

Note that this implementation of PPO uses separate networks for critics
and actors.

# MAPPO

This implementation of PPO supports multi-agent environments (MAPPO), and,
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

Design decisions that may have an impact on learning have largely come
from the following two papers:
arXiv:2103.01955v2
arXiv:2006.07869v4

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


# Installation

After activating a virtual environment, issue the following command from the
top directory.
```
pip install .
```

NOTE: required packages are not yet listed.

# Environments

## Gymnasium

Both single agent and multi-agent gymnasium games are supported through
the `SingleAgentGymWrapper` and `MultiAgentGymWrapper`, respectively.
For examples on how to train a gymnasium environment, see the following
scripts:
* `baselines/lunar_lander.py` for single agent training.
* `baselines/pressure_plate.py` for multi agent training. NOTE: this
   environment is also wrapped with `Gym21ToGymnasium`, which is only
   needed for environments that only exist in gym versions <=21.

## Gym <= 0.21

For environments that only exist in versions <= 0.21 of Gym, you
can use the `Gym21ToGymnasium` wrapper. See `baselines/pressure_plate.py`
for an example.

## Gym To Gymnasium

Games that exist in Gym versions >= 0.26 but not Gymnasium can be tricky.
I've found that the biggest issue is the spaces not matching up. We have
a function `gym_space_to_gymnasium_space` in `environments/gym/version_wrappers.py`
that can be used to (attempt to) convert spaces from Gym to Gymnasium.

## Abmarl

The `AbmarlWrapper` can be used for Abmarl environments. See `baselines/abmarl_maze.py`
for an example.

## Petting Zoo

The `ParallelZooWrapper` can be used for Abmarl environments. See `baselines/abmarl_maze.py`
for an example.

## Custom

All environments must be wrapped in the `PPOEnvironmentWrapper`. If you're
using a custom environment that doesn't conform to Gym or Abmarl standards,
you can create your own wrapper that inherits from `PPOEnvironmentWrapper`,
found in `environments/ppo_env_wrappers.py`.

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
            "lr"               : lr,
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
ppoaf --train <path_to_runner_file>
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
ppoaf <path_to_runner_file> --num-test-runs <num_test_runs> --render
```
You can optionally omit the `--render` or add the `--render-gif` flag.

By default, exploration is disabled during testing, but you can enable it
with the `--test-explore` flag. Example:

```
ppoaf <path_to_runner_file> --num-test-runs <num_test_runs> --render --test-explore
```

Note that enabling exploration during testing will have varied results. I've found
that most of the environments I've tested perform better without exploration, but
there are some environments that will not perform at all without it.


# Documentation

When specific implentation choices have been made as a result of a publication,
the goal is to reference these publications in the relevant code. For
implementation details that are not directly derived from publications, there
should be appropriate comments describing why these choices were made.

Documentation is a work in progress.

# MPI And Environments Per Processor
Both MPI and multiple environment instances per processor are supported,
and utilizing these options can greatly speed up training time. Some
environments may be sensitive to the choices here, which can have an
impact on training. See the **Tips and Tricks** section for some general
setting suggestions.

Currently, the default is to use GPUs when training on a single processor
and CPUs when training on multiple processors. This can be overridden with
the `--alow-mpi-gpu` flag, which is helpful for environments that require
networks that can benefit from GPUs (convolutions, for example).

NOTE: the current implementation of multiple environment instances per
processor assumes that the rollout bottleneck will come from inference rather
than stepping through the environment. Because of this, the multiple environment
instances are run in succession rather than in parallel, and the speed up
comes from batched inference during the rollout. Very slow environments may
not see a performance gain from increasing `envs-per-proc`.

**Usage:**

mpirun:
```
mpirun -n {num_procs} ppoaf --envs-per-proc {envs_per_proc} ...
```

srun:
```
srun -N1 -n {num_procs} ppoaf --envs-per-proc {envs_per_proc} ...
```

Some things to note:
1. The total timesteps per rollout is divided among the processors. So,
   if the environment is set to run 1024 timesteps per rollout, each
   processor will collect 1024/N of those timesteps, where N is the total
   number of processors (remainders go to processor 0). Note that there
   are various side effects of this, some of which are outlined below.
   Also, `envs-per-proc` can have a similar effect on reducing the total
   timesteps that each environment instance experiences, especially if
   each instance can reach its max timesteps before being "done".
2. Increasing the processor count doesn't always increase training speed.
   For instance, imagine an environment that can only reach unique states
   in the set `U` after running for at least 500 time steps. If our total
   timesteps per rollout is set to 1024, and we run with > 2 processors,
   we will never collect states from `U` and thus might not ever learn
   how to handle those unique situations. A similar logic applies for
   `envs-per-proc`. **Note**: this particular issue is now partially
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

# Other Features

## Observational Augmentations

### What are observational augmentations?
Imagine we have a 3D environment where we
want to learn to move a ball as fast as possible. There are two options to
consider here; do we want the ball to move in a single direction, or should
the ball be able to move in any direction? In other words, should direction
matter at all in our reward? If the answer is yes, we should learn to move
the ball in one direction only, then we can proceed as usual. Environments
like Ant employ this kind of reward. However, if our answer is no, we have
an interesting situation here. If we're using the standard approach for this
case, we need to wait for the policy to experience enough directions to
learn that direction doesn't matter. On the other hand, the policy might
learn a simplified version of this idea early on and just choose a random
direction and stick with it. But what if we *want* the policy to learn that
direction doesn't matter? In this case, we really do need to wait for the
policy to encounter enough experiences to learn this on its own (maybe we start
the ball rolling in a random direction with every reset). If direction truly
doesn't matter, though, do we really need to wait for those experiences
to stack up? Well, this likely depends on the environment. In this case,
our observations are simple enough that we can *augment* a single observation
to obtain a batch of observations corresponding to different directions.
We can get away with this because *the reward trajectories will be identical
regardless of direction*. In other words, we can cheat the system, and,
instead of waiting for the environment to crank out a bunch of directions,
we can take a short cut by artificially generating those directions and
their trajectories from a single observation.

### How to utilize observational augmentations
Both `run`, located in `runners/env_runner.py` and our `PPO` class have
an `obs_augment` argument that, when enabled, will try to wrap your environment
in `AugmentingEnvWrapper`. This wrapper expects (and checks) that your
environment has a method named `augment_observation`, which takes a *single*
observation and returns a batch of observations. This batch will contain the
original observation plus a number of augmentations, and the batch size is
expected to always be the same. The values for actions, dones, and rewards
are all duplicated to be identical for every instance in the batch. Info is also
duplicated, except that it might contain augmented terminal observations.

**NOTE:** we don't currently prohibit the use of multiple environments per
processor in conjunction with `aug_observation`, but it is untested and
should be used with caution and consideration.

## Soft Resets

### What are soft resets?
In short, the environment is only reset back to its starting state when it
reaches a done state. This can be useful when you want to keep your timesteps
per episode fairly short while allowing your agent(s) to explore the
environment at further time states.

### When to use caution
While soft resets can be very useful, there are also situations where they
can be detrimental. Imagine a scenario where your agent can easily fall into
inescapable "traps" midway through exploring the environment. If soft resets
are enabled, you might find that your rollouts are starting with the agent
in this trap, which could negatively impact learning. On the other hand, if
the traps are escapable, and escaping is a desired learned behavior, then
using soft resets might actually be helpful in the long term.

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

**OpenAI Gym**

Installing atari environments:
```
pip install gym[atari]
pip install autorom[accept-rom-license]
```

Mujoco sometimes requires some extra tweaks. There is a `mujoco_export.sh` file
that can help with some of these issues. For testing with the `--render` flag,
you'll need to set the `LD_PRELOAD` path (see the above bash file). For running,
with the `--render-gif` flag, you'll need to unset the `LD_PRELOAD` path.

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

## Environment Settings

I have by no means run any comprehensive studies to figure out what the
optimal choices are for hyper-parameters, but the settings located
in the baselines tend to work well. In general, using as
many processors as you have access to will speed up training, but
there is also communication overhead that will become more noticable
as you scale up. This, combined with the limitations of how often
the policy is updated, will lead to diminishing returns at some point.

The number of environments per processor is a little less straightforward,
as this setting will further divide up the episode lengths. I typically see
good results by setting this to something between 2 and 4, depending on the
environment.

I've added extra information and tips for some of our more unique environments
below.

### BipedalWalker
The environment is considered solved when the average score over 100 test 
runs is >= 300. This policy easily gets an average score >= 320.

### BipedalWalkerHardcore
This is one of the most challenging environments in this repo, and it takes a
significant amount of time to reach a solved policy. Using 4 processors and
1 environment per processor, I generally see a solved policy around 4 hours
or so of training (~6000->7000 iterations), but the policy can still be a 
bit brittle at this point. For example, testing the policy 4 times in a row
with each test averaging over 100 test runs might result in 3 out of the 4
averages being >= 300. Longer training will result in a more stable policy.
The environment is considered solved when the average score over 100 test
runs is >= 300. This policy generally gets an average score >= 320 once
solved.

### All Atari pixel environments
I recommend using `--device 'gpu'` for systems with GPUs.

### Ant
In order to solve the environment, you need to reach an average score >= 6000
over 100 test runs.

### Walker2d and Hopper
Both Walker2d and Hopper are oddly sensitive to the trajectory lengths.
They will learn to run a bit beyond the max trajectory length very well,
but they often stumble afterwards. For this reason, I recommend using no
training with a single environment per processor.
This will result in a solved policy fairly quickly.

### HumanoidStandup
This is an unsolved environment.

### RobotWarehouseTiny
There are many configuration options for the robot warehouse. This one
uses 3 agents in a "tiny" warehouse.
This environment has very sparse rewards, so it can take a while for the
agents to explore enough to reach a good policy.

### RobotWarehouseSmall
There are many configuration options for the robot warehouse. This one
uses 4 agents in a "small" warehouse.
This is the same as RobotWarehouseTiny, except that it is slightly larger. The
complexity of the environment is increased with the increase in size, so
learning takes longer.

### LevelBasedForaging
There are many configuration options for this environment. This configuration
uses 3 players, an 8x8 grid, 2 food sources, and each agent is aware of
the entire grid.

### PressurePlate
This environment allows configuration of the number of players. This
configuration uses 4 players.

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

# PPO And Friends

PPO and Friends is a PyTorch implementation of Proximal Policy Optimation
along with various extra optimizations and add-ons (freinds).

While this project supports many of OpenAI's gym environments, the goal
is to be fairly independent of gym. Because of this, you'll often see
situations where utilities that gym provides have been ignored in favor of
creating our own, often simplified (or sometimes more complicated...), versions
of these utilities. This is largely to support custom environments that might
not follow the standard rules that gym enforces.

# Our Friends

Some of our friends:

* Intrinsic Curiosity Module (ICM)
* Generalized Advantage Estimations (GAE)
* LSTM integration into PPO algorithm
* Gradient, reward, bootstrap, and observation clipping
* KL based early ending
* Splitting observations by proprioceptive and exteroceptive information
* Observation, advantage, and reward normalization
* Learning rate annealing
* Vectorized environments
* Observational augmentations

# Installation

After activating a virtual environment, issue the following command from the
top directory.
```
pip install .
```

NOTE: required packages are not yet listed.

# Supported Environments

Supported environments and a general idea of good training settings can
be viewed in the **Tips and Tricks** section of this README. A full list
can also be viewed by issuing the following command:
```
python main.py --help
```

To train an already supported environment, use the following command:
```
python main.py -e <env_name> --num_timesteps <max_num_timesteps>
```

Running the same command again will result in loading the previously
saved state. You can re-run from scratch by using the `--clobber` option.

To test a model that has been trained on a particular environment,
you can issue the following command:
```
python main.py -e <env_name> --num_test_runs <num_test_runs> --render
```
You can optionally omit the `--render` or add the `--render_gif` flag.


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
the `--alow_mpi_gpu` flag, which is helpful for environments that require
networks that can benefit from GPUs (convolutions, for example).

**Usage:**

mpirun:
```
mpirun -n {num_procs} python main.py -e {env_name} --envs_per_proc {envs_per_proc} ...
```

srun:
```
srun -N1 -n {num_procs} python main.py -e {env_name} --envs_per_proc {envs_per_proc} ...
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
   mitigated by the recent addition of "soft resets".
3. When running with multiple processors or environment instances,
   the stats that are displayed might not fully reflect the true status
   of learning. For instance, imagine an environment that, when performing
   well, receives +1 for every timestep and is allowed to run for a
   maximum of 100 timesteps. This results in a max score of +100. If each
   processor is only collecting 32 timesteps per rollout, the highest
   score any of them could ever achieve would be 32. Therefore, a reported
   score around 32 might actually signal a converged policy.

# Observational Augmentations

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
Both `run_ppo`, located in `environments/launchers.py` and our `PPO` class have
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

# Tips And Tricks

**OpenAI Gym**

Installing atari environments:
```
pip install gym[atari]
pip install autorom[accept-rom-license]
```

Mujoco sometimes requires some extra tweaks. There is a `mujoco_export.sh` file
that can help with some of these issues. For testing with the `--render` flag,
you'll need to set the `LD_PRELOAD` path (see the above bash file). For running,
with the `--render_gif` flag, you'll need to unset the `LD_PRELOAD` path.

**Performance**

Unless otherwise stated, all provided environments should be capable of
training "successful" policies. When it's provided, the definition of
success comes from the environment itself. BipedalWalker, for instance, defines
success as reaching an average score >= 300 over 100 test iterations. For
environments that do not provide definitions of success, I wing it.

Currently, the only "unsolved" environment in this repository is the
HumanoidStandup environment.

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
optimal choices are for processor and `env_per_proc` distribution, but
I've found some general settings that tend to work well. I've outlined
them below. I've included rough timing information for environments that
I remembered to record.

### CartPole
4 processors and 2 environments per processor works well here. An excellent
policy will be learned ~20->30 iterations, which takes a matter of seconds.

### Pendulum
2 processors and 2 environments per processor works well here. I see an
excellent policy after ~7->12 minutes of training.

### Acrobot
2 procssors and 1 environment per processor will learn a good policy in
roughly 30 seconds to a minute.

### MountainCar
2 processors and 3 environments per processor takes about 6-10 minutes
to solve the environment.

### MountainCarContinuous
2 processors and 3 environments per processor works well here.

### LunarLander
2 processors and 2 environments per processor works well here. I see an
excellent policy in under 2 minutes (~400 iterations).

### LunarLanderContinuous
2 processors and 2 environments per processor works well here. I see an
excellent policy in under 5 minutes (~100 iterations).

### BipedalWalker
2 processors and 1 environments per processor will solve the environment
in about 15->20 minutes. The environment is considered solved when the
average score over 100 test runs is >= 300. This policy easily gets
an average score >= 320.

### BipedalWalkerHardcore
4 processors and 1 environments per processor works well. This is probably
the most challenging solveable environment in this repo, and it takes a
significant amount of time to reach a solved policy. I generally see a solved
policy around 4 hours or so of training (~6000->7000 iterations), but the
policy can still be a bit brittle at this point. For example, testing the
policy 4 times in a row with each test averaging over 100 test runs might
result in 3 out of the 4 averages being >= 300. Longer training will result
in a more stable policy.
The environment is considered solved when the average score over 100 test
runs is >= 300. This policy generally gets an average score >= 320 once
solved.

### All Atari pixel environments
I recommend enabling the `--allow_mpi_gpu` flag for systems with GPUs. I
tested BreakoutPixels using 4 processors and 2 environments per processor,
which worked well.

### All Atari RAM environments
I tested BreakoutRAM using 4 processors and 2 environments per processor,
which worked well.

### InvertedPendulum
2 processors and 2 environments per processor learns an excellent policy in
roughly 10 seconds of training.

### InvertedDoublePendulum
2 processors and 2 environments per processor learns a good policy within
a few minutes.

### Ant
2 processors and 2 environments per processor learns a excellent policy within
10 minutes of training.

### Walker2d
Both Walker2d and Hopper are oddly sensitive to the trajectory lengths.
They will learn to run a bit beyond the max trajectory length very well,
but they often stumble afterwards. For this reason, I recommend using no
more than 2 processors and training with a single environment per processor.
This will result in a solved policy fairly quickly.

### Hopper
Both Walker2d and Hopper are oddly sensitive to the trajectory lengths.
They will learn to run a bit beyond the max trajectory length very well,
but they often stumble afterwards. For this reason, I recommend using no
more than 2 processors and training with a single environment per processor.
This will result in a solved policy fairly quickly.

### Swimmer
2 processors and 2 environments per processor works very well.

### HalfCheetah
2 processors and 2 environments per processor learns an excellent policy
in about 2 minutes.

### Humanoid
2 processors and 2 environments per processor learns an excellent policy
within 40 minutes or less.

### HumanoidStandup
Who knows with this one...

# Resulting Policies
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

## LunarLander
<img src="https://github.com/aowen87/ppo_and_friends/blob/main/gifs/LunarLander.gif" width="300" height="200" />

- **test score: 259.4**

## LunarLanderContinuous
<img src="https://github.com/aowen87/ppo_and_friends/blob/main/gifs/LunarLanderContinuous.gif" width="300" height="200" />

- **test score: 281.7**

## BipedalWalker
<img src="https://github.com/aowen87/ppo_and_friends/blob/main/gifs/BipedalWalker.gif" width="300" height="200" />

- **test score: 326.2**
- **averge over 100 test runs: ~319**

## BipedalWalkerHardcore
<img src="https://github.com/aowen87/ppo_and_friends/blob/main/gifs/BipedalWalkerHardcore.gif" width="300" height="200" />

- **test score: 329.7**
- **averge over 100 test runs: ~313**

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

- **test score: 3651.3**

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

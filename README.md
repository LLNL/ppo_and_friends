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
* Gradient, reward, bootstrap, and observation clipping
* KL based early ending
* Splitting observations by proprioceptive and exteroceptive information
* Observation, advantage, and reward normalization
* Learning rate annealing

# Installation

After activating a virtual environment, issue the following command from the
top directory.
```
pip install .
```

NOTE: required packages are not yet listed.

# Supported Environments

Environments that are currently implemented can be found in
environments/launchers.py. A full list can also be viewed by issuing
the following command:
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
You can optionally omit the `--render` flag.


# Documentation

When specific implentation choices have been made as a result of a publication,
the goal is to reference these publications in the relevant code. For
implementation details that are not directly derived from publications, there
should be appropriate comments describing why these choices were made.

Documentation is a work in progress.

# MPI
MPI training is supported and can greatly speed up the training process.
Currently, GPUs are only used when training on a single processor, and
CPUs are used when training on more than one processor.

**Usage:**

mpirun:
```
mpirun -n {num_procs} python main.py ...

```

srun:
```
srun -n {num_procs} python main.py ...

```

Some things to note:
1. The total timesteps per rollout is divided among the processors. So,
   if the environment is set to run 1024 timesteps per rollout, each
   processor will collect 1024/N of those timesteps, where N is the total
   number of processors (remainders go to processor 0). Note that there
   are various side effects of this, some of which are outlined below.
2. Increasing the processor count doesn't always increase training speed.
   For instance, imagine an environment that can only reach unique states
   in the set `U` after running for at least 500 time steps. If our total
   timesteps per rollout is set to 1024, and we run with > 2 processors,
   we will never collect states from `U` and thus might not ever learn
   how to handle those unique situations. **Note**: this particular issue
   is now partially mitigated by the recent addition of "soft resets".
3. When running with multiple processors, the stats that are displayed
   might not fully reflect the true status of learning. For instance,
   imagine an environment that, when performing well, receives +1 for
   every timestep and is allowed to run for a maximum of 100 timesteps.
   This results in a max score of +100. If each processor is only collecting
   32 timesteps per rollout, the highest score any of them could ever
   achieve would be 32. Therefore, a reported score around 32 might
   actually signal a converged policy.


# Tips And Tricks

**OpenAI Gym**

Installing atari environments:
```
pip install gym[atari]
pip install autorom[accept-rom-license]
```

Mujoco sometimes requires some extra tweaks. There is a `mujoco_export.sh` file
that can help with some of these issues.

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

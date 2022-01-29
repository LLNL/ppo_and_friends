# PPO And Friends

PPO and Friends is a PyTorch implementation of Proximal Policy Optimation
along with various extra optimizations and add-ons (freinds).

While this project supports many of OpenAI's gym environments, the goal
is to be fairly independent of gym. Because of this, you'll often see
situations where utilities that gym provides have been ignored in favor of
creating our own, often simplified, versions of these utilities. This is
largely to support custom environments that might not follow the standard
rules that gym enforces.

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

After activating a virtuale environment, issue the following command from the
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

Note that, even though random seeds are set, Torch is still allowed to use
stochastic algorithms. This results in training performance varying from
one run to another.


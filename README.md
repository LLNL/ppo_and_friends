# PPO And Friends

PPO And Friends (PPO-AF) is an MPI distributed PyTorch implementation of
Proximal Policy Optimation along with various extra optimizations and
add-ons (freinds).

We are currently compatible with the following environment frameworks:
* Gymnasium
* Gym (including versions <= 0.21)
* PettingZoo
* Abmarl Gridworld

## Our Friends

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

# Getting Started

## Installation

While you can install a barebones version of PPO-AF by simply issuing a
`pip install .` command, most situations will require installing one of our
supported RL library extensions:

1. gym (version 0.21.0): `pip install .[gym]`
2. gymnasium: `pip install .[gymnasium]`
3. abmarl: `pip install .[abmarl]`
4. petting zoo: `pip install .[pettingzoo]`

Installing the gym extension may also require downgrading your pip wheel:
```
pip install --upgrade pip wheel==0.38.4
```

## Environment Runners

To train an environment, an [EnvironmentRunner](./runners/env_runner.py) must first be defined. The
runner will be a class that inherits from
[EnvironmentRunner](./runners/env_runner.py) or the [GymRunner](./runners/env_runner.py).
located within the same module. The only method you need to define is
`run`, which should call `self.run_ppo(...)`.

[See this CartPole example](./baselines/gymnasium/cart_pole.py)

**Make note of the following requirements**:
1. your environment MUST be wrapped in one of the available ppo-and-friends
   environment wrappers. Currently available wrappers are SingleAgentGymWrapper,
   MultiAgentGymWrapper, AbmarlWrapper, and ParallelZooWrapper. See
   [Environment Wrappers](#environment-wrappers) for more info.
2. You must add the [@ppoaf_runner](./runners/runner_tags.py) decorator to your class.

See the `baselines` directory for more examples.

## Training

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

### MPI

PPO-AF is designed to work seamlessly with MPI. To train across multiple ranks
and nodes, simply issue your MPI command followed by the PPO-AF command.

**Examples:**

mpirun:
```
mpirun -n {num_procs} ppoaf ...
```

srun:
```
srun -N1 -n {num_procs} ppoaf ...
```

### Environments Per Processor

The current implementation of multiple environment instances per
processor assumes that the rollout bottleneck will come from inference rather
than stepping through the environment. Because of this, the multiple environment
instances are run in succession rather than in parallel, and the speed up
comes from batched inference during the rollout. Very slow environments may
not see a performance gain from increasing `envs_per_proc`.

**Examples:**

mpirun:
```
mpirun -n {num_procs} ppoaf --envs_per_proc {envs_per_proc} ...
```

srun:
```
srun -N1 -n {num_procs} ppoaf --envs_per_proc {envs_per_proc} ...
```

## Evaluating

### Testing Trained Policies
To test a model that has been trained on a particular environment,
you can issue the following command:
```
ppoaf test <path_to_output_directory> --num_test_runs <num_test_runs> --render
```

By default, exploration is enabled during testing, but you can disable it
with the `--deterministic` flag. Example:

```
ppoaf test <path_to_output_directory> --num_test_runs <num_test_runs> --render --deterministic
```
The output directory will be given the same name as your runner file, and
it will appear in the path specified by `--state_path` when training, which
defaults to `./saved_states`.

### Plotting Results
If `--save_train_scores` is used while training, the results can be plotted using
PPO-And-Friend's ploting utility.

```
ppoaf plot path1 path2 path3 ... <options>
```

# Terminology
Terminology varies across implemenations and publications, so here are
some commonly overloaded terms and how we define them.

1. **batch size**: we refer to the gradient descent mini-batch size as the
   batch size. This is sometimes referred to as 'mini batch size',
   'sgd mini batch size', etc. This is defined as `batch_size` in our code.
2. **timesteps per rollout**: this refers to the total number of timesteps
   collected in a single rollout. This is sometimes referred to as the batch
   size. This is defined on a per-environment per-processor basis, i.e.
   `ts_per_rollout` will be internally redefined as
   `ts_per_rollout = (num_procs * ts_per_rollout * envs_per_proc) / num_procs`.
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
wrappers are provided in the [environments](./environments) directory. For best results,
all environments should be wrapped in a class inherting from
[PPOEnvironmentWrapper](./environments/ppo_env_wrappers).

## PPO
This is the default policy for single-agent environments.

## MAPPO and IPPO
arXiv:2103.01955v4 makes the distinction between MAPPO and IPPO such that
the former uses a centralized critic receiving global information about
the agents of a shared policy (usually a concatenation of the observations),
and the later uses an independent, decentralized critic.

Both options can be enabled by setting the `critic_view` parameter in
the [PPOEnvironmentWrapper](./environments/ppo_env_wrappers) appropriately. Options as of now are
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

All multi-agent environment wrappers that inherit from [PPOEnvironmentWrapper](./environments/ppo_env_wrappers) 
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
the [SingleAgentGymWrapper](./environments/gym/wrappers.py) and [MultiAgentGymWrapper](./environments/gym/wrappers.py), respectively.
For examples on how to train a gymnasium environment, check out the runners
in [baselines/gymnasium/](./baselines/gymnasium).

**IMPORTANT**: While Gymnasium does not have a standard interface for multi-agent games,
I've found some commonalities among many publications, and we are using this
as our standard. You may need to make changes to your multi-agent gymnasium
environments before they can be wrapped in the [MultiAgentGymWrapper](./environments/gym/wrappers.py).

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
can use the [Gym21ToGymnasium](./environments/gym/version_wrappers.py) wrapper. See [baselines/gym/](./baselines/gym)
for examples.

**IMPORTANT**: While Gym does not have a standard interface for multi-agent games,
I've found some commonalities among many publications, and we are using this
as our standard. You may need to make changes to your multi-agent gymnasium
environments before they can be wrapped in the [MultiAgentGymWrapper](./environments/gym/wrappers.py).

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
a function [gym_space_to_gymnasium_space](./environments/gym/version_wrappers.py)
that can be used to (attempt to) convert spaces from Gym to Gymnasium.

## Abmarl

The [AbmarlWrapper](./environments/abmarl/wrappers.py) can be used for Abmarl environments. See [baselines/abmarl](./baselines/abmarl) for
examples.

## Petting Zoo

The [ParallelZooWrapper](./environments/petting_zoo/wrappers.py) can be used for PettingZoo environments. See [baselines/pettingzoo](./baselines/petting_zoo)
for examples.

## Custom

All environments must be wrapped in the [PPOEnvironmentWrapper](./environments/ppo_env_wrappers.py). If you're
using a custom environment that doesn't conform to supported standards,
you can create your own wrapper that inherits from [PPOEnvironmentWrapper](./environments/ppo_env_wrappers.py).

# Authors

PPO-AF was created by Alister Maguire, maguire7@llnl.gov.

# Contributing

PPO-AF is open source, and contributing is easy.
1. Create a branch with your changes.
2. Make sure that your changes work. Add tests if appropriate.
3. Open a pull request and add a reviewer.

# License

The code of this site is released under the MIT License. For more details, see the [LICENSE](LICENSE) File.

LLNL-CODE-867112

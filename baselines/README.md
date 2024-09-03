# Baselines

The `baselines` directory contains a number of pre-defined `EnvironmentRunners`
that can be used as references.

Policies can differ from one training to another, and the longer training
sessions generally result in better policies. For the results demonstrated
below, I trained for a moderate amount, which is usually just enough to
see a decent policy. See the **Environment Setttings** section for details.

## CartPole
<img src="https://github.com/LLNL/ppo_and_friends/blob/main/gifs/CartPole.gif" width="300" height="200" />

- **test score: 200**
- **average over 100 test runs: 200**

## Pendulum
<img src="https://github.com/LLNL/ppo_and_friends/blob/main/gifs/Pendulum.gif" width="300" height="200" />

- **test score: -241.6**

## Acrobot
<img src="https://github.com/LLNL/ppo_and_friends/blob/main/gifs/Acrobot.gif" width="300" height="200" />

- **test score: -82**

## MountainCar
<img src="https://github.com/LLNL/ppo_and_friends/blob/main/gifs/MountainCar.gif" width="300" height="200" />

- **test score: -108**
- **average over 100 test runs: -105.1**

## MountainCarContinuous
<img src="https://github.com/LLNL/ppo_and_friends/blob/main/gifs/MountainCarContinuous.gif" width="300" height="200" />

- **test score: 94.6**
- **average over 100 test runs: 92.0**

## LunarLander
<img src="https://github.com/LLNL/ppo_and_friends/blob/main/gifs/LunarLander.gif" width="300" height="200" />

- **test score: 259.4**

## LunarLanderContinuous
<img src="https://github.com/LLNL/ppo_and_friends/blob/main/gifs/LunarLanderContinuous.gif" width="300" height="200" />

- **test score: 281.7**

## BipedalWalker
<img src="https://github.com/LLNL/ppo_and_friends/blob/main/gifs/BipedalWalker.gif" width="300" height="200" />

- **test score: 326.2**
- **average over 100 test runs: ~319**

## BipedalWalkerHardcore
<img src="https://github.com/LLNL/ppo_and_friends/blob/main/gifs/BipedalWalkerHardcore.gif" width="300" height="200" />

- **test score: 329.7**
- **average over 100 test runs: ~313**

## BreakoutRAM
<img src="https://github.com/LLNL/ppo_and_friends/blob/main/gifs/BreakoutRAM.gif" width="300" height="200" />

- **test score: N/A**: I cut off the test at 500 steps for this gif.

## InvertedPendulum
<img src="https://github.com/LLNL/ppo_and_friends/blob/main/gifs/InvertedPendulum.gif" width="300" height="200" />

- **test score: 1000**

## InvertedDoublePendulum
<img src="https://github.com/LLNL/ppo_and_friends/blob/main/gifs/InvertedDoublePendulum.gif" width="300" height="200" />

- **test score: 9318.5**

## Ant
<img src="https://github.com/LLNL/ppo_and_friends/blob/main/gifs/Ant.gif" width="300" height="200" />

- **test score: 6106.2**
- **average over 100 test runs: 6298.3**

## Walker2d
<img src="https://github.com/LLNL/ppo_and_friends/blob/main/gifs/Walker2d.gif" width="300" height="200" />

- **test score: 3530.0**

## Hopper
<img src="https://github.com/LLNL/ppo_and_friends/blob/main/gifs/Hopper.gif" width="300" height="200" />

- **test score: 3211.0**

## Swimmer
<img src="https://github.com/LLNL/ppo_and_friends/blob/main/gifs/Swimmer.gif" width="300" height="200" />

- **test score: 131.3**

## HalfCheetah
<img src="https://github.com/LLNL/ppo_and_friends/blob/main/gifs/HalfCheetah.gif" width="300" height="200" />

- **test score: 4157.9**

## Humanoid
<img src="https://github.com/LLNL/ppo_and_friends/blob/main/gifs/Humanoid.gif" width="300" height="200" />

- **test score: 6330.9**

## RobotWarehouseTiny
<img src="https://github.com/LLNL/ppo_and_friends/blob/main/gifs/RobotWarehouseTiny.gif" width="300" height="200" />

- **test score (averaged across all agents): 11.0**

## LevelBasedForaging
<img src="https://github.com/LLNL/ppo_and_friends/blob/main/gifs/LevelBasedForaging.gif" width="300" height="200" />

- **test score (averaged across all agents): 0.25**
- **highest test score (max across all agents): 0.33**

## PressurePlate
<img src="https://github.com/LLNL/ppo_and_friends/blob/main/gifs/PressurePlate.gif" width="300" height="200" />

- **test score (averaged across all agents): -19.27**

## MPESimpleAdversary
<img src="https://github.com/LLNL/ppo_and_friends/blob/main/gifs/MPESimpleAdversary.gif" width="300" height="200" />

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
<img src="https://github.com/LLNL/ppo_and_friends/blob/main/gifs/MPESimpleTag.gif" width="300" height="200" />

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
<img src="https://github.com/LLNL/ppo_and_friends/blob/main/gifs/MPESimpleSpread.gif" width="300" height="200" />

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

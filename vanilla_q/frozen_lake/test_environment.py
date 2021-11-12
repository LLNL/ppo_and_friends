import numpy as np
import time
import gym

env = gym.make("FrozenLake-v1")
env.reset()

#
# The ice can be slippery, meaning that the agent/player can
# randomly slip into an unintended direction.
#
print("Initial state:")
env.render()
print()

print()
print("Taking action 0")
env.reset()
env.step(0)
env.render()
print()

print()
print("Taking action 1")
env.reset()
env.step(1)
env.render()
print()

print()
print("Taking action 2")
env.reset()
env.step(2)
env.render()
print()

print()
print("Taking action 3")
env.reset()
env.step(3)
env.render()
print()


env.close()

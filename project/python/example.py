#!/usr/bin/env python3

"""
Example script to test the Inverted Double Pendulum environment.
"""

###########
# Imports #
###########

import gym
import pybulletgym
import time


########
# Main #
########

# Initialize

env = gym.make('InvertedDoublePendulumPyBulletEnv-v0')

env.render()
env.reset()

# Camera

# /!\ Only work with pybulletgym personal modifications!
env.camera.env._p.resetDebugVisualizerCamera(2, 0, -20, [0, 0, 0])

# Parameters

n_episodes = 10
max_steps = 50

timestep = 1 / 30

# Episodes

for _ in range(n_episodes):
    observation = env.reset()

    ## Steps

    for _ in range(max_steps):
        env.render()

        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)

        time.sleep(timestep)

        if done:
            break

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  7 21:08:16 2020

@author: shashank
"""

from gym_torcs import TorcsEnv
import numpy as np

#### Generate a Torcs environment
# enable vision input, the action is steering only (1 dim continuous action)
env = TorcsEnv(vision=True, throttle=False)

# without vision input, the action is steering and throttle (2 dim continuous action)
# env = TorcsEnv(vision=False, throttle=True)

ob = env.reset(relaunch=True)  # with torcs relaunch (avoid memory leak bug in torcs)
# ob = env.reset()  # without torcs relaunch

# Generate an agent
# from sample_agent import Agent
# agent = Agent(1)  # steering only

# action = agent.act(ob, reward, done, vision=True)

# single step
for i in range(5000):
    ob, reward, done, _ = env.step(np.random.random(1))
    # a = make_vec(ob)
    print(ob)

# shut down torcs
env.end()
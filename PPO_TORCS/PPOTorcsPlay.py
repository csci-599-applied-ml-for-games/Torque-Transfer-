#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 12:36:54 2020

@author: shashank
"""
from lib.Model import ActorCritic
from gym_torcsDum import TorcsEnv

import torch

model_name = '/home/shashank/Desktop/Coursework/Sem2/AMLG/TORCS/PPO_TORCS/checkpoints/TORCS_best_+418412.996_307200.dat'
HIDDEN_SIZE = 256


def test_env(env, model, device, deterministic=True):
    state = env.reset()
    done = False
    total_reward = 0
    i = 0
    while (not done) and (i<10240):
        # env.render()
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        dist, _ = model(state)
        action = dist.mean.detach().cpu().numpy()[0] if deterministic \
            else dist.sample().cpu().numpy()[0]
        next_state, reward, done, _ = env.step(action)
        state = next_state
        total_reward += reward
        i +=1
    return total_reward

# Autodetect CUDA
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
print('Device:', device)

# Prepare environments
env = TorcsEnv(vision=False, throttle=False, gear_change=False)
# num_inputs = env.observation_space.shape[0]
num_inputs = 71
num_outputs = env.action_space.shape[0]

model = ActorCritic(num_inputs, num_outputs, HIDDEN_SIZE).to(device)
model.load_state_dict(torch.load(model_name))

print(model)

test_env(env, model, device)
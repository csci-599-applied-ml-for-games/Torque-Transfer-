import os
import gym
import gym_donkeycar
import numpy as np

exe_path = '/home/tushar/Projects/donkeycar_new/Donkey Car Simulator/DonkeySimLinux/donkey_sim.x86_64'
port = 9091
print(os.path.exists('/home/tushar/Projects/donkeycar_new/DonkeySimLinux/donkey_sim.x86_64'))
exit()
env = gym.make("donkey-warehouse-v0", exe_path=exe_path, port=port)

obv = env.reset()
for t in range(100):
    action = np.array([0.0,0.5]) # drive straight with small speed
# execute the action
obv, reward, done, info = env.step(action)
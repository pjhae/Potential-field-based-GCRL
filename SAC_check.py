import gym
from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.vec_env import VecFrameStack
import sys

date = "0617"
trial = "C"
steps = "1860000"

# First trial for fixed maze
# date = "0615"
# trial = "A"
# steps = "1040000"

# GOOD FOR no obstacle env
# date = "0614"
# trial = "C"

# steps = "350000"

## Make gym environment #
env = make_vec_env("Point-v3", n_envs=1)

## Path ##
save_path='./save_model_'+date+'/'+trial+'/'

## Load Model##
model = SAC.load(save_path+"Point_model_"+date+trial+"_"+steps+"_steps", device='cuda')


## Rendering ##

obs = env.reset()
while True:
    action, _states = model.predict(obs,deterministic=True)
    obs, rewards, done, info = env.step(action)
    env.render()

f.close()


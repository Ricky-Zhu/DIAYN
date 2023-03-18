import gym
from goal_env.mujoco import *

env = gym.make('AntEmpty-v0')
s=env.reset()

for i in range(1000):
    s,r,d,_=env.step(env.action_space.sample())
    env.render()

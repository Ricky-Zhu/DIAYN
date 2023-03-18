import gym
from gym import Wrapper
import numpy as np
from goal_env.mujoco import *

class WrapperDictEnv(Wrapper):
    def __init__(self,env):
        super().__init__(env)
        self.observation_space = self.observation_space.spaces['observation']

    def step(self, action):
        obs, r, done, info = self.env.step(action)
        s = obs['observation']
        r = 0.

        if info['is_success']:
            r = 1.
        done = False

        return s, r, done, info

    def reset(self, **kwargs):
        s = self.env.reset()['observation']
        return s



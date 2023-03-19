import gym


env = gym.make('Hopper-v2')
s=env.reset()

for i in range(1000):
    s,r,d,_=env.step(env.action_space.sample())
    env.render()

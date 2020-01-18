"""
https://www.youtube.com/watch?v=6Yd5WnYls_Y
"""

from ddpg import Agent
import gym
import numpy as np
from show_plot import plotLearning

env = gym.make('LunarLanderContinuous-v2')

agent = Agent(alpha=0.000025, beta=0.000025, input_dims=[8], tau=0.001, env=env,
              batch_size=64, layer1_size=400, layer2_size=300, n_actions=2)

np.random.seed(0)

score_history = []

for i in range(1000):
    done = False
    score = 0
    obs = env.reset()
    while not done:
        act = agent.choose_action(obs)
        new_state, reward, done, info = env.step(act)
        agent.remember(obs, act, reward, new_state, int(done))
        agent.learn()
        score += reward
        obs = new_state

    score_history.append(score)
    print('episode ', i, 'score %.3f' % score,
          '100 game average %.3f' % np.mean(score_history[-100:]))
    if i % 25 == 0:
        agent.save_models()

filename = 'lunar-lander.png'
plotLearning(score_history, filename, window=100)

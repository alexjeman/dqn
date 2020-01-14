import gym
import numpy as np
import matplotlib.pyplot as plt

from agent import Agent

if __name__ == '__main__':
    env = gym.make('FrozenLake-v0')

    lr = 0.001
    gamma = 0.9
    n_actions = 4
    n_states = 16
    eps_start = 1.0
    eps_end = 0.01
    eps_dec = 0.9999995
    n_games = 500000

    agent = Agent(lr, gamma, n_actions, n_states, eps_start, eps_end, eps_dec)

    scores = []
    win_pct_list = []

    for i in range(n_games):
        done = False
        observation = env.reset()
        score = 0
        while not done:
            action = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            agent.learn(observation, action, reward, observation_)
            score += reward
            observation = observation_
        scores.append(score)
        if i % 100 == 0:
            win_pct = np.mean(scores[-100:])
            win_pct_list.append(win_pct)
            if i % 1000 == 0:
                print(f"episode: {i}, win percentage: {win_pct:.2f}, epsilon: {agent.epsilon:.3f}")
    plt.plot(win_pct_list)
    plt.show()

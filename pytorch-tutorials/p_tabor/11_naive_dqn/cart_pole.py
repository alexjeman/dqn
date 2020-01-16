import gym
import numpy as np
import matplotlib.pyplot as plt
from show_plot import plot_learning_curve

from naive_dqn import Agent

if __name__ == '__main__':
    env = gym.make('CartPole-v1')
    input_dims = env.observation_space.shape
    n_actions = env.action_space.n

    lr = 0.0001
    gamma = 0.99
    epsilon = 1.0
    eps_min = 0.01
    eps_dec = 1e-5
    n_games = 10000

    scores = []
    eps_history = []

    agent = Agent(input_dims, n_actions, lr, gamma, epsilon, eps_dec, eps_min)

    print(agent.Q.device)

    for i in range(n_games):
        score = 0
        done = False
        state = env.reset()                                     # Reset done flag and env to it's initial state

        while not done:
            action = agent.choose_action(state)                 # Choose action according to epsilon greedy action selection using current state of the env as input
            state_, reward, done, info = env.step(action)       # Get new state reward done and debug info from new env after taking that action
            score += reward                                     # Increment score by the reward
            agent.learn(state, action, reward, state_)          # Next learn from from state, action, reward and new state
            state = state_                                      # Set the old state to new state so in the next state we are choosing a action from correct new state of the env
        scores.append(score)
        eps_history.append(agent.epsilon)

        if i % 100 == 0:
            avg_score = np.mean(scores[-100:])
            print(f"episode: {i}, score: {score}, avg score: {avg_score:.2f}, epsilon: {agent.epsilon:.3f}")
    filename = 'naive_dqn.png'
    x = [i + 1 for i in range(n_games)]
    plot_learning_curve(x, scores, eps_history, filename)

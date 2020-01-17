import numpy as np
from agent import DQNAgent
from env_preprocessor import make_env
from show_plot import plot_learning_curve

if __name__ == '__main__':
    env = make_env('PongNoFrameskip-v4')
    env_name = 'PongNoFrameskip-v4'
    chkpt_dir = 'models/'
    input_dims = env.observation_space.shape
    n_actions = env.action_space.n

    best_score = -np.inf
    algo='DQNAgent'
    render_game = False
    load_checkpoint = True
    train_model = True
    n_games = 500
    gamma = 0.99
    epsilon = 1.0
    lr = 1e-4
    eps_min = 1e-5
    eps_dec = 1e-5
    replace = 1000
    mem_size = 20000
    batch_size = 32

    agent = DQNAgent(gamma=gamma, epsilon=epsilon, lr=lr, input_dims=input_dims,
                     n_actions=n_actions, mem_size=mem_size, eps_min=eps_min,
                     batch_size=batch_size, replace=replace, eps_dec=eps_dec, chkpt_dir=chkpt_dir, algo=algo,
                     env_name=env_name)

    if load_checkpoint:
        agent.load_models()
        agent.epsilon = eps_min

    fname = agent.algo + '_' + agent.env_name + '_lr' + str(agent.lr) + '_' \
            '_' + str(n_games) + 'games'
    figure_file = 'plots/' + fname + '.png'
    scores_file = fname + '_scores.npy'

    n_episodes = 0
    scores = []
    eps_history = []
    episode_arr = []

    for i in range(n_games):
        done = False
        observation = env.reset()
        score = 0

        while not done:
            if render_game:
                env.render()

            action = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            score += reward

            if train_model:
                agent.store_transition(observation, action, reward, observation_, int(done))
                agent.learn()
            observation = observation_

        scores.append(score)

        avg_score = np.mean(scores[-100:])
        print('episode: ', i, ',score: ', score, ',average score: %.1f ,best score: %.1f epsilon %3f' % (avg_score, best_score, agent.epsilon))

        if avg_score > best_score:
            best_score = avg_score
            if train_model and i > 50:
                agent.save_models()

        eps_history.append(agent.epsilon)
    n_episodes += 1
    episode_arr.append(n_episodes)

    plot_learning_curve(episode_arr, scores, eps_history, figure_file)

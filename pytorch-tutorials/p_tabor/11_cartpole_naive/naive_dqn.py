import gym  # Gym env
import numpy as np  # To handle averaging of prev 100 games
import torch as T
import torch.nn as nn  # Gives access to the layers
import torch.nn.functional as F  # Gives access to activation functions
import torch.optim as optim  # Gives access to optimizers


class LinearDeepQNetwork(nn.Module):                                        # Derivates fron nn.Module
    def __init__(self, lr, n_actions, input_dims):
        super(LinearDeepQNetwork, self).__init__()

        self.fc1 = nn.Linear(*input_dims, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)               # Define optimizer
        self.loss = nn.MSELoss()                                            # Define loss function
        self.device = T.device("cuda" if T.cuda.is_available() else "cpu")  # Detect if CUDA GPU is available
        self.to(self.device)                                                # Send NN to GPU

    def forward(self, state):
        layer1 = F.relu(self.fc1(state))                                    # Get the output from layer1, forward to layer2, 3 and activate, note the order
        layer2 = F.relu(self.fc2(layer1))
        layer3 = F.relu(self.fc3(layer2))
        actions = self.fc4(layer3)                                          # Not activating last layer, Loss function will handle this for us

        return actions                                                      # Return action


class Agent():
    def __init__(self, input_dims, n_actions, lr, gamma, epsilon, eps_dec, eps_min):
        self.lr = lr
        self.input_dims = input_dims
        self.n_actions = n_actions
        self.action_space = [i for i in range(self.n_actions)]               # List of actions in space from 0 to n
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_dec = eps_dec
        self.eps_min = eps_min

        self.Q = LinearDeepQNetwork(self.lr, self.n_actions, self.input_dims)  # Q value for our agent, agent has Q estimate, Agent is not a Q estimate

    def choose_action(self, observation):                                      # Chooses an observation of the environment as input
        if np.random.random() > self.epsilon:                                  # Calculate random value for epsilon greedy selection, if it is greater than epsilon take greedy action, otherwise take random action
            state = T.tensor(observation, dtype=T.float).to(self.Q.device)
            actions = self.Q.forward(state)                                    # Get values of action state and find argmax
            action = T.argmax(actions).item()                                  # Find argmax by taking T.argmax and converting to NP item for actual value from tensor
        else:
            action = np.random.choice(self.action_space)                       # Get random action if np.random.random() is less than current epsilon greedy
        return action

    def decrement_epsilon(self):                                               # Reduce epsilon random action by decreasing epsilon, aprox 1/4 from total number of the games
        self.epsilon = self.epsilon - self.eps_dec \
                        if self.epsilon > self.eps_min else self.eps_min

    def learn(self, state, action, reward, state_):
        self.Q.optimizer.zero_grad()                                             # First thing to do at every learning function is to zero gradients between loops
        states = T.tensor(state, dtype=T.float).to(self.Q.device)              # Convert arrays to Pytorch CUDA tensors
        actions = T.tensor(action).to(self.Q.device)
        rewards = T.tensor(reward).to(self.Q.device)
        states_ = T.tensor(state_, dtype=T.float).to(self.Q.device)

        q_pred = self.Q.forward(states)[actions]                               # Calculate feed forward to get our Q estimate, get predicted values for current state of the environment, delta between action agent took and the maximum action he could have been taken in that particular state
                                                                               # Target is the maximum proffitable action for the given state and the distance of where we are and that target is the value given by the action agent actually took
                                                                               # So we want to get action indicies from this q_pred-icted tensor

        q_next = self.Q.forward(states_).max()                                 # In calculation of the target value the maximum action for the agent estimate of the resulting state

        q_target = rewards + self.gamma*q_next                                  # Target/direction we want to move in is going to be the reward+GAMMA*Value of the maximum action in the next state

        cost = self.Q.loss(q_target, q_pred).to(self.Q.device)                 # Cost is going to be the difference/delta/distance between action agent actually took and the maximum value agent could have been taken (mean suqared error of that loss)
        cost.backward()                                                        # Back propagate the cost and optimizer function, this is where dqn learns
        self.Q.optimizer.step()
        self.decrement_epsilon()                                               # Decrement epsilon greedy - take less random actions as game keeps going

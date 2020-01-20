import os
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np


class DoubleDeepQNetwork(nn.Module):
    def __init__(self, lr, n_actions, name, input_dims, chkpt_dir):
        super(DoubleDeepQNetwork, self).__init__()
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name)

        self.conv1 = nn.Conv2d(input_dims[0], 32, kernel_size=8, stride=4)  # input_dims the [0]'th element corresponds to the number of channels in the input image(in this case 4 by (1 gray scale image)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        fc_input_dims = self.calculate_conv_output_dims(input_dims)

        self.fc1 = nn.Linear(fc_input_dims, 512)
        self.fc2 = nn.Linear(512, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.SmoothL1Loss()
        self.device = T.device("cuda" if T.cuda.is_available() else "cpu")
        self.to(self.device)

    def calculate_conv_output_dims(self, input_dims):
        state = T.zeros(1, *input_dims)
        dims = self.conv1(state)
        dims = self.conv2(dims)
        dims = self.conv3(dims)
        return int(np.prod(dims.size()))

    def forward(self, state):
        conv1 = self._swish(self.conv1(state))
        conv2 = self._swish(self.conv2(conv1))
        conv3 = self._swish(self.conv3(conv2))
        conv_state = conv3.view(conv3.size()[0], -1)
        flat1 = self._swish(self.fc1(conv_state))
        flat2 = self._swish(self.fc2(flat1))
        flat3 = self._swish(self.fc3(flat2))
        actions = self.fc4(flat3)

        return actions

    def _swish(self, x):
        return x * T.sigmoid(x)

    def save_checkpoint(self):
        print('Saving checkpoint...')
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        print('Loading checkpoint...')
        self.load_state_dict(T.load(self.checkpoint_file))

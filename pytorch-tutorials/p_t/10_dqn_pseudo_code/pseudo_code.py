import torch
import torch.nn as nn  # Gives access to the layers
import torch.nn.functional as F  # Gives access to activation functions
import torch.optim as optim  # Gives access to optimizers


class LinearClassifier(nn.Module):
    def __init__(self, lr, n_classes, input_dims):
        super(LinearClassifier, self).__init__()

        # Define NN Layers
        self.fc1 = nn.Linear(*input_dims, 128) # 3 layer NN with Linear activation function
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, n_classes)

        # Define optimizer and loss function
        self.optimizer = optim.Adam(self.parameters(), lr=lr)  # Optimizer Adam
        self.loss = nn.MSELoss()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Detect if Cuda GPU is available
        self.to(self.device)  # Send NN to GPU

    def forward(self, data):  # Pytorch handles back propagation, but we need to define forward activation function
        layer1 = F.sigmoid(self.fc1(self.fc1(data)))  # Pass data to the first fully connected conned layer and activate with F.sigmoid
        layer2 = F.sigmoid(self.fc2(layer1))  # Get the output from layer 1, forward to layer 2 and activate, note the order
        layer3 = self.fc3(layer2)  # Not activating last layer, Loss function will handle this for us
        return layer3

    # Define learning loop
    def learn(self, data, labels):
        self.optimizer.zero_grad() # First thing to do at every learning function is to zero gradients between loops
        data = torch.tensor(data).to(self.device)  # Convert data to tensors on device assuming it was not done so
        labels = torch.tensor(data).to(self.device) # Same for labels

        predictions = self.forward(data)  # Get feed forward for data

        cost = self.loss(predictions, labels)  # Get cost associated with how far away are those predictions from the actual labels

        cost.backward()  # Back propagate the cost and optimizer function, this is where dqn learns
        self.optimizer.step()

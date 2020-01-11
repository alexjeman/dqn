import numpy as np


class neural_network(object):
    def __init__(self):
        # Define Hyperparameters
        self.inputLayers = 2
        self.hiddenLayers = 3
        self.outputLayers = 1

        # Weights (Parameters)
        self.W1 = np.random.randn(self.inputLayers, self.hiddenLayers)
        print(self.W1)
        self.W2 = np.random.randn(self.hiddenLayers, self.outputLayers)
        print(self.W2)

    def forward(self, X):
        # Propagate inputs trough network
        self.z2 = np.dot(X, self.W1)
        self.a2 = self.activationf(self.z2)
        self.z3 = np.dot(self.a2, self.W2)
        y = self.activationf(self.z3)
        return y

    def activationf(self, z):
        # Apply sigmoid activation function to scalar, vector, or matrix
        return 1/(1+np.exp(-z))
        # Apply ReLu activation function
        # return np.maximum(0, z)


x = [2, 5]

NN = neural_network()

output = NN.forward(x)

print(output)

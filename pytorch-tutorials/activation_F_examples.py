import numpy as np

y = np.random.random()

def sigmoid(X):
    return 1 / (1 + np.exp(-X))

def relu(X):
    return np.maximum(0.0, X)

print(sigmoid(y))
print(relu(y))

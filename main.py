import numpy as np
import nnfs
from nnfs.datasets import spiral_data

nnfs.init()

class Layer_Dense:  
    def __init__(self, n_inputs, n_neurons):
        # Initialize weights and biases.
        # Weights are (inputs, neurons) rather than (neurons, inputs)
        # This means we don't have to  transpose on every pass (discussed in Chapter 2)
        # np.random.randn produces a Gaussian distribution with a mean of 0 and variance of 1.
        # We multiply by 0.01 to reduce the numbers as otherwise the model will take more time
        # to fit the data during training and the starting values will be disproprtionately large.
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        # Initialize a row vector of zeroes for weights.
        self.biases = np.zeros((1, n_neurons))
    
    def forward(self, inputs):
        # Calculate output values from inputs, weights and biases.
        self.output = np.dot(inputs, self.weights) + self.biases

# ReLU Activation
class Activation_ReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)

# Create dataset
X, y = spiral_data(samples = 100, classes = 3)

# Create Dense Layer with 2 input features & 3 output values (neurons).
dense1 = Layer_Dense(2, 3)

# Create ReLU activation to be used with Dense Layer
activation1 = Activation_ReLU()

# Perform Forward Pass of the training data through this layer.
dense1.forward(X)

# Forward pass through activation func.
# Uses output from previous layer.
activation1.forward(dense1.output)

print(activation1.output[:5])

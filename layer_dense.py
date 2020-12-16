import numpy as np
import nnfs

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

nnfs.init()

layer = Layer_Dense(2, 4)
print(layer.weights, layer.biases)
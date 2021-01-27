import numpy as np

# Passed-in gradient from the next layer
# for the purpose of this example we're going to use
# an array of an incremental gradient values
dvalues = np.array([[1., 1., 1.],
                    [2., 2., 2.],
                    [3., 3., 3.]])

# We have 3 sets of weights - one set for each neuron
# we have 4 inputs, thus 4 weights
# recall that we keep weights transposed
weights = np.array([[0.2, 0.8, -0.5, 1],
                    [0.5, -0.91, 0.26, -0.5],
                    [-0.26, -0.27, 0.17, 0.87]]).T

# Weights are transposed so the layer receives derivatives with respect to inputs
# As opposed to neuron weights.
#print(weights)

# Sum weights related to the given input multiplied by
# the gradient related to the given neuron
# dx0 = sum([weights[0][0]*dvalues[0][0],
#            weights[0][1]*dvalues[0][1],
#            weights[0][2]*dvalues[0][2]])
#dx0 = sum(weights[0]*dvalues[0])
#dinputs = np.array([dx0, dx1, dx2, dx3])

dinputs = np.dot(dvalues, weights.T)

# dinputs is a gradient of the neuron function with respect to inputs.
print("Derivative with respect to Inputs")
print(dinputs)

#DERIVATIVE WITH RESPECT TO WEIGHTS:

# We have 3 sets of inputs - samples
inputs = np.array([[1, 2, 3, 2.5],
                   [2., 5., -1., 2],
                   [-1.5, 2.7, 3.3, -0.8]])
# sum weights of given input
# and multiply by the passed-in gradient for this neuron
# Derivative of a multiplication (input * weight) is the coefficient
# With respect to weights - this is input.
dweights = np.dot(inputs.T, dvalues)
print("Derivative with respect to Weights")
print(dweights)

#DERIVATIVE WITH RESPECT TO BIASES:
# Derivative of a sum is always 1 & the bias is a sum.
# Multiply 1 by subsequent function due to chain rule.

# One bias for each neuron
# biases are the row vector with a shape (1, neurons)
biases = np.array([[2, 3, 0.5]])
# dbiases - sum values, do this over samples (first axis), keepdims
# since this by default will produce a plain list -
# we explained this in the chapter 4
dbiases = np.sum(dvalues, axis=0, keepdims=True)
print("Derivative with respect to Biases")
print(dbiases)

# DERIVATIVE WITH RESPECT TO ReLU:
# Example layer output
z = np.array([[1, 2, -3, -4],
              [2, -7, -1, 3],
              [-1, 2, 5, -1]])
dvalues = np.array([[1, 2, 3, 4],
                    [5, 6, 7, 8],
                    [9, 10, 11, 12]])
# ReLU activation's derivative
print("Derivative with respect to ReLU")
# Produce array of 0s in same shape as z.
# drelu = np.zeros_like(z)
# drelu[z > 0] = 1
# Simplified:
drelu = dvalues.copy()
drelu[z <= 0] = 0

print(drelu)
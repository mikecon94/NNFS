import numpy as np

# Passed-in gradient from the next layer
# for the purpose of this example we're going to use
# a vector of 1s
dvalues = np.array([[1., 1., 1.]])

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

dinputs = np.dot(dvalues[0], weights.T)

# dinputs is a gradient of the neuron function with respect to inputs.
print(dinputs)

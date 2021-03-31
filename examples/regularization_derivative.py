# Derivatives of weights on a single neuron.
# 1 if positive.
# -1 if negative.
weights = [0.2, 0.8, -0.5] # weights of one neuron
dL1 = [] # array of partial derivatives of L1 regularization
for weight in weights:
 if weight >= 0:
    dL1.append(1)
 else:
    dL1.append(-1)
print(dL1)

# Derivatives for multiple neurons in a layer
weights = [[0.2, 0.8, -0.5, 1], # now we have 3 sets of weights
           [0.5, -0.91, 0.26, -0.5],
           [-0.26, -0.27, 0.17, 0.87]]
dL1 = [] # array of partial derivatives of L1 regularization
for neuron in weights:
    neuron_dL1 = [] # derivatives related to one neuron
    for weight in neuron:
        if weight >= 0:
            neuron_dL1.append(1)
        else:
            neuron_dL1.append(-1)
    dL1.append(neuron_dL1)
print(dL1)

# Numpy version:
import numpy as np

weights = np.array([[0.2, 0.8, -0.5, 1],
                    [0.5, -0.91, 0.26, -0.5],
                    [-0.26, -0.27, 0.17, 0.87]])

dL1 = np.ones_like(weights)
dL1[weights < 0] = -1
print(dL1)
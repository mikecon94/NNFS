import numpy as np

inputs = [1.0, 2.0, 3.0, 2.5]
weights = [[0.2, 0.8, -0.5, 1.0],
	  [0.5, -0.91, 0.26, -0.5],
	  [-0.26, -0.27, 0.17, 0.87]]
biases = [2.0, 3.0, 0.5]

# np.dot() - hanldes the 2D array and returns a vector of the same size
# (3 in this case)
# np.dot(weights, inputs) = [np.dot(weights[0], inputs), np.dot(weights[1], inputs) etc.]
# Whatever comes first in np.dot will determine the output shape.
# np.dot((3, 4), 4) = output vector of (3)
outputs = np.dot(weights, inputs) + biases

print(outputs)

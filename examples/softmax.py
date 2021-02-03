import math
import numpy as np

# Values from previous layer's output
layer_outputs = [4.8, 1.21, 2.385]

# calculate exponential value for each value in vector.
exp_values = np.exp(layer_outputs)
print("Exponentiated Values:", exp_values)

# Normalize the values
# 
norm_values = exp_values / np.sum(exp_values)

print("Normalized exponentiated values:", norm_values)
print("Sum of normalized values:", np.sum(norm_values))

# With Batches of inputs:
layer_outputs = np.array([[4.8, 1.21, 2.385],
                          [8.9, -1.81, 0.2],
                          [1.41, 1.051, 0.026]])
# Axis tells np what to sum and changes return shape
# Default is None - sum up all values across dimensions.
print('Sum without axis:', np.sum(layer_outputs))
print('Sum with axis=None (default):', np.sum(layer_outputs, axis=None))
# 0 means to sum row-wise (along axis 0).
#   For each position of output Sum values from all the other dimensions at this position.
#   Columns for a matrix. 4.8 + 8.9 + 1.41 etc.
print('Sum with axis=0:', np.sum(layer_outputs, axis=0))

# 1 Means to sum the rows instead which is what we want.
print('Sum with axis=1:', np.sum(layer_outputs, axis=1))
# Gave the sums as expected but shape is (0, 3)
# We need a single value per sample (3, 1). Column Vector.
# This will let us normalize the whole batch of samples, sample-wise, with a single calculation.
# (we need to sum all the outputs from a layer for each sample in a batch)
# keepdims=True keeps the input dimensions.
print(np.sum(layer_outputs, axis=1, keepdims=True))

# We also subtract the largest of the inputs before the exponentiation.
# Dead neurons & exploding values are 2 big challenges with neural networks.
# Exponential function is a source of exploding values - could cause an overflow error.
# exp tends toward 0 as input approaches negative infinity.
# Output is 1 when input is 0.
print(np.exp(-np.inf), np.exp(0))
# We can use this to prevent an overflow.
# Subtract maximum value from a list of inputs.
# This would change outputs to be in a range from some negative up to 0.
# max - max = 0.
# We can do this subtraction thanks to normalization which prevents this operation changing the output.
np.exp(layer_outputs - np.max(layer_outputs, axis=1, keepdims=True))

class Activation_Softmax:
    def forward(self, inputs):
        # Get unnormalized probabilities
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        # Normalize them for each sample
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities

softmax = Activation_Softmax()
softmax.forward([[1, 2, 3]])
print(softmax.output)
# Subtract max (3) from each value and get the same result.
softmax.forward([[-2, -1, 0]])
print(softmax.output)



# BACKWARDS PASS - PAGE 226

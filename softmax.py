import math
import numpy as np

# Values from previous layer's output
layer_outputs = [4.8, 1.21, 2.385]

# calculate exponential value for each value in vector.
exp_values = np.exp(layer_outputs)
print("Exponentiated Values:", exp_values)

# Normalize the values
norm_values = exp_values / sum(exp_values)

print("Normalized exponentiated values:", norm_values)
print("Sum of normalized values:", np.sum(norm_values))
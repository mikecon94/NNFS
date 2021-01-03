import math

# Values from previous output
layer_outputs = [4.8, 1.21, 2.385]

# e - math constant.
E = math.e

# calculate exponential value for each value in vector.
exp_values = []
for output in layer_outputs:
    exp_values.append(E ** output)
print("Exponentiated Values:", exp_values)

# Normalize the values
norm_base = sum(exp_values)
norm_values = []

for value in exp_values:
    norm_values.append(value / norm_base)

print("Normalized exponentiated values:", norm_values)
print("Sum of normalized values:", sum(norm_values))
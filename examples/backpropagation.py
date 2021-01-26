# We will start with a single neuron ReLU Activation

x = [1.0, -2.0, 3.0] # input values
w = [-3.0, -1.0, 2.0] # weights
b = 1.0 # bias

# Multiply input by the weight
xw0 = x[0] * w[0]
xw1 = x[1] * w[1]
xw2 = x[2] * w[2]
print("Weighted Inputs:", xw0, xw1, xw2)

# Next sum of all weighted inputs with a bias
z = xw0 + xw1 + xw2 + b
print("Sum of Weights Inputs & Bias:", z)

# Apply ReLU
y = max(z, 0)
print("ReLU:", y)

# Backward Pass

# Derivative from next layer
dvalue = 1.0

# Derivative of ReLU and the chain rule
drelu_dz = dvalue * (1. if z > 0 else 0.)
print("Derivative of ReLU:", drelu_dz)

# Partial derivative of the sum wrt the x input
# weighted, for the 0th pair of inputs and weights.
# 1 is the value of this partial derivative, which we multiply
# using hte chain rule. Weight the derivative of the subsequent function:
# The ReLU function.
dsum_dxw0 = 1
drelu_dxw0 = drelu_dz * dsum_dxw0
print("Partial Derivative of sum function wrt 1st weighted input:", drelu_dxw0)

# We then do the same operation with the next weighted inputs:
dsum_dxw1 = 1
drelu_dxw1 = drelu_dz * dsum_dxw1
print("Partial Derivative of sum function wrt 2nd weighted input:", drelu_dxw1)

dsum_dxw2 = 1
drelu_dxw2 = drelu_dz * dsum_dxw1
print("Partial Derivative of sum function wrt 3rd weighted input:", drelu_dxw2)

# Now the bias
dsum_db = 1
drelu_db = drelu_dz * dsum_db
print("Partial Derivative of sum function wrt bias:", drelu_db)

# Continuing backwards. Function before sum is multiplication of weights and inputs.
dmul_dx0 = w[0]
drelu_dx0 = drelu_dxw0 * dmul_dx0
print("Partial Derivative of the multiplication wrt x0:", drelu_dx0)

# Repeat for other inputs and weights
dmul_dx0 = w[0]
dmul_dx1 = w[1]
dmul_dx2 = w[2]
dmul_dw0 = x[0]
dmul_dw1 = x[1]
dmul_dw2 = x[2]
drelu_dx0 = drelu_dxw0 * dmul_dx0
drelu_dw0 = drelu_dxw0 * dmul_dw0
drelu_dx1 = drelu_dxw1 * dmul_dx1
drelu_dw1 = drelu_dxw1 * dmul_dw1
drelu_dx2 = drelu_dxw2 * dmul_dx2
drelu_dw2 = drelu_dxw2 * dmul_dw2
print(drelu_dx0, drelu_dw0, drelu_dx1, drelu_dw1, drelu_dx2, drelu_dw2)
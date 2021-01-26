# We will start with a single neuron ReLU Activation

x = [1.0, -2.0, 3.0] # input values
w = [-3.0, -1.0, 2.0] # weights
b = 1.0 # bias

# Multiply input by the weight
xw0 = x[0] * w[0]
xw1 = x[1] * w[1]
xw2 = x[2] * w[2]
print(xw0, xw1, xw2)

# Next sum of all weighted inputs with a bias
z = xw0 + xw1 + xw2 + b
print(z)

# Apply ReLU
y = max(z, 0)
print(y)
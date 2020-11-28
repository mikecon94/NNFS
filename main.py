# 4 Inputs
# 3 Neurons (3 sets of weights, 3 biases)
# Example Layer

inputs = [1, 2, 3, 2.5]
# Input 1 has weight of 0.2 etc.
weights = [[0.2, 0.8, -0.5, 1.0],
	  [0.5, -0.91, 0.26, -0.5],
	  [-0.26, -0.27, 0.17, 0.87]]

# 1 bias per neuron.
biases = [2, 3, 0.5]

# Output of layer
layer_outputs = []

# zip() pairs the arrays together and returns an iterator
# Allows us to iterate over multiple iterables (lists) simultaneously.
# For each neuron
for neuron_weights, neuron_bias in zip(weights, biases):
	# Reset output value to 0 for each neuron.	
	neuron_output = 0
	# For each input & weight to the neuron	
	for n_input, weight in zip(inputs, neuron_weights):
		# Multiple input by associated weight
		# Add to output variable.
		neuron_output += n_input * weight
	# Add Bias
	neuron_output += neuron_bias
	layer_outputs.append(neuron_output)

print(layer_outputs)




inputs = [1, 2, 3]
# Input 1 has weight of 0.2 etc.
weights = [0.2, 0.8, -0.5]
# 1 bias per neuron.
bias = 2

#This neuron sums each input multiplied by the input weight. Then adds the bias.
output = (inputs[0]*weights[0] +
	inputs[1]*weights[1] +
	inputs[2]*weights[2] + bias)
#Outputs 2.3
print(output)


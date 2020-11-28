# 4 Inputs
# 3 Neurons (3 sets of weights, 3 biases)
# Example Layer

inputs = [1, 2, 3, 2.5]
# Input 1 has weight of 0.2 etc.
weights1 = [0.2, 0.8, -0.5, 1.0]
weights2 = [0.5, -0.91, 0.26, -0.5]
weights3 = [-0.26, -0.27, 0.17, 0.87]

# 1 bias per neuron.
bias1 = 2
bias2 = 3
bias3 = 0.5

#This neuron sums each input multiplied by the input weight. Then adds the bias.
outputs = [
	#Neuon 1:	
	(inputs[0]*weights1[0] +
	inputs[1]*weights1[1] +
	inputs[2]*weights1[2] +
	inputs[3]*weights1[3] + bias1),

	#Neuron 2:
	(inputs[0]*weights2[0] +
	inputs[1]*weights2[1] + 
	inputs[2]*weights2[2] +
	inputs[3]*weight2[3] + bias2),

	#Neuron 3:
	(inputs[0]*weights3[0] +
	inputs[1]*weights3[1] +
	inputs[2]*weights3[2] +
	inputs[3]*weights3[3] + bias3)]

print(outputs)


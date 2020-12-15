import numpy as np

#Batch of inputs (matrix).
#Shape - (3, 4)
inputs = [[1.0, 2.0, 3.0, 2.5],
	  [2.0, 5.0, -1.0, 2.0],
	  [-1.5, 2.7, 3.3, -0.8]]
#Shape - (3,4)
# Each set of weights is a neuron.
weights = [[0.2, 0.8, -0.5, 1.0],
	  	   [0.5, -0.91, 0.26, -0.5],
	  	   [-0.26, -0.27, 0.17, 0.87]]
biases = [2.0, 3.0, 0.5]

weights2 = [[0.1, -0.14, 0.5],
			[-0.5, 0.12, -0.33],
			[-0.44, 0.73, -0.13]]

biases2 = [-1, 2, -0.5]

#Matrix Product requires the no. of rows of matrix 1 (3)
#to match no. of columns of matrix 2 (4)
#Index 1 of matrix 1.
#Index 0 of matrix 2.
#We transpose so they do (1 weight per input)
#Matrix Product - multiply each value in row of matrix 1
#by each value in column of matrix 2.
layer1_outputs = np.dot(inputs, np.array(weights).T) + biases
layer2_outputs = np.dot(layer1_outputs, np.array(weights2).T) + biases2

#Output is a list of layer outputs per sample.
#This is why inputs is first parameter.
#Sample related output so it can be passed into the next layer.
print(layer2_outputs)

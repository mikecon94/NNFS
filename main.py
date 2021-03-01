import numpy as np
import nnfs
from nnfs.datasets import spiral_data
from nnfs.datasets import vertical_data

nnfs.init()

class Layer_Dense:  
    def __init__(self, n_inputs, n_neurons):
        # Initialize weights and biases.
        # Weights are (inputs, neurons) rather than (neurons, inputs)
        # This means we don't have to  transpose on every pass (discussed in Chapter 2)
        # np.random.randn produces a Gaussian distribution with a mean of 0 and variance of 1.
        # We multiply by 0.01 to reduce the numbers as otherwise the model will take more time
        # to fit the data during training and the starting values will be disproprtionately large.
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        # Initialize a row vector of zeroes for weights.
        self.biases = np.zeros((1, n_neurons))
    
    def forward(self, inputs):
        # Calculate output values from inputs, weights and biases.
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases

    def backward(self, dvalues):
        # Gradients on parameters
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        # Gradient on Values / Inputs
        self.dinputs = np.dot(dvalues, self.weights.T)

# ReLU Activation
class Activation_ReLU:
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.maximum(0, inputs)

    def backward(self, dvalues):
        # Gradient with respect to inputs on ReLU.
        # Copy next layer and zero out values where the input was less than 0.
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= 0] = 0

class Activation_Softmax:
    def forward(self, inputs):
        # Get unnormalized probabilities
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        # Normalize them for each sample
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities
    
    def backward(self, dvalues):
        # Create uninitialized array
        self.dinputs = np.empty_like(dvalues)

        # Enumerate outputs and gradients
        for index, (single_output, single_dvalues) in \
                enumerate(zip(self.output, dvalues)):
            # Flatten Output Array
            single_output = single_output.reshape(-1, 1)
            # Calculate Jacobian matrix of the output
            jacobian_matrix = np.diagflat(single_output) - \
                              np.dot(single_output, single_output.T)
            # Calculate sample-wise gradient
            # and add it to the array of sample gradients
            self.dinputs[index] = np.dot(jacobian_matrix, single_dvalues)

# Base/Common Loss Class
class Loss:
    # Calculates the data and regularization losses.
    # given model output and ground truth values.
    def calculate(self, output, y):
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses)
        return data_loss

# Cross Entropy Loss
class Loss_CategoricalCrossEntropy(Loss):
    def forward(self, y_pred, y_true):
        # Number of samples in a batch.
        samples = len(y_pred)

        # Clip data at both sides to prevent division by 0
        # And avoid dragging the mean towards a value.
        y_pred_clipped = np.clip(y_pred, 1e-7, 1-1e-7)

        # Probabilities for target values:
        # If categorical labels
        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[
                                    range(samples),
                                    y_true   
                                  ]
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_pred_clipped * y_true, axis=1)

        # Losses
        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods
    
    def backward(self, dvalues, y_true):
        # Number of samples
        samples = len(dvalues)
        
        # Number of labels in every sample
        # We'll use the first sample to count them
        labels = len(dvalues[0])

        # If labels are sparse, turn into one-hot vector
        if len(y_true.shape) == 1:
            y_true = np.eye(labels)[y_true]
        
        # Calculate gradient
        self.dinputs = -y_true /dvalues
        # Normalize gradient
        self.dinputs = self.dinputs / samples
        return 1

# Softmax Classifier - Combined Softmax activation
# with cross-entropy loss for faster backward step.
class Activation_Softmax_Loss_CategoricalCrossEntropy():
    def __init__(self):
        self.activation = Activation_Softmax()
        self.loss = Loss_CategoricalCrossEntropy()
    
    def forward(self, inputs, y_true):
        self.activation.forward(inputs)
        self.output = self.activation.output
        # Calculate & Return loss value
        return self.loss.calculate(self.output, y_true)
    
    def backward(self, dvalues, y_true):
        # No. of samples
        samples = len(dvalues)
        # If labels are one-hot encoded
        # Turn them into discrete values
        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)
        
        # Copy so we can modify
        self.dinputs = dvalues.copy()
        # Calculate gradient
        self.dinputs[range(samples), y_true] -= 1
        # Normalize gradient
        self.dinputs = self.dinputs / samples

# Create dataset
# X is coordinates
# y is the class
X, y = spiral_data(samples = 100, classes = 3)

# Create Dense Layer with 2 input features & 3 output values (neurons).
dense1 = Layer_Dense(2, 3)

# Create ReLU activation to be used with Dense Layer
activation1 = Activation_ReLU()

# Create second Dense layer with 3 input features (output of previous layer)
# and 3 output values.
dense2 = Layer_Dense(3, 3)

# Create Softmax activation (use with dense2 layer)
activation2 = Activation_Softmax()

# Define loss function
loss_function = Loss_CategoricalCrossEntropy()

# Perform Forward Pass of the training data through this layer.
dense1.forward(X)

# Forward pass through activation func.
# Takes output of first dense layer.
activation1.forward(dense1.output)

# Forward pass through second Dense layer.
# Takes output of activation function from layer 1.
dense2.forward(activation1.output)

# Forward pass through activation2 function (softmax) 
# Takes output of 2nd dense layer as input.
activation2.forward(dense2.output)

print(activation2.output[:5])

# Loss Calculation
# Forward pass through loss function
# Takes output of second dense layer and returns loss.
loss = loss_function.calculate(activation2.output, y)
print("Loss:", loss)

# Accuracy

predictions = np.argmax(activation2.output, axis=1)
if len(y.shape) == 2:
    y = np.argmax(y, axis=1)
accuracy = np.mean(predictions == y)
print("Accuracy:", accuracy)



# Comparing Backpropagation of Softmax Activation with 
# CC Entropy loss combined
softmax_outputs = np.array([[0.7, 0.1, 0.2],
                            [0.1, 0.5, 0.4],
                            [0.02, 0.9, 0.08]])
class_targets = np.array([0, 1, 1])

softmax_loss = Activation_Softmax_Loss_CategoricalCrossEntropy()
softmax_loss.backward(softmax_outputs, class_targets)
combinedLoss = softmax_loss.dinputs

activation = Activation_Softmax()
activation.output = softmax_outputs
loss = Loss_CategoricalCrossEntropy()
loss.backward(softmax_outputs, class_targets)
activation.backward(loss.dinputs)
separateLoss = activation.dinputs
print('Gradients: combined loss and activation:')
print(combinedLoss)
print('Gradients: separate loss and activation:')
print(separateLoss)
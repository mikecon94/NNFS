# THIS WAS REPLACED BY THE regression.py file in the root.
# This was then changed to nnfs.py at the end of Chapter 19.
# This file support classification problems.

import numpy as np
import nnfs
from nnfs.datasets import spiral_data

nnfs.init()

# Different versions of Numpy/Python produce different outputs.
# NNFS book uses:
# Python 3.7.5
# Numpy 1.15.0
# Matplotlib 3.1.1

class Layer_Dense:  
    def __init__(self, n_inputs, n_neurons,
                 weight_regularizer_l1=0, weight_regularizer_l2=0,
                 bias_regularizer_l1=0, bias_regularizer_l2=0):
        # Initialize weights and biases.
        # Weights are (inputs, neurons) rather than (neurons, inputs)
        # This means we don't have to  transpose on every pass (discussed in Chapter 2)
        # np.random.randn produces a Gaussian distribution with a mean of 0 and variance of 1.
        # We multiply by 0.01 to reduce the numbers as otherwise the model will take more time
        # to fit the data during training and the starting values will be disproprtionately large.
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        # Initialize a row vector of zeroes for weights.
        self.biases = np.zeros((1, n_neurons))
        # Self Regularization Strength
        self.weight_regularizer_l1 = weight_regularizer_l1
        self.weight_regularizer_l2 = weight_regularizer_l2
        self.bias_regularizer_l1 = bias_regularizer_l1
        self.bias_regularizer_l2 = bias_regularizer_l2
    
    def forward(self, inputs):
        # Calculate output values from inputs, weights and biases.
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases

    def backward(self, dvalues):
        # Gradients on parameters
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)

        # Gradients on regularization
        #L1 on weights
        if self.weight_regularizer_l1 > 0:
            dL1 = np.ones_like(self.weights)
            dL1[self.weights < 0] = -1
            self.dweights  += self.weight_regularizer_l1 * dL1
        
        #L2 on weights
        if self.weight_regularizer_l2 > 0:
            self.dweights += 2 * self.weight_regularizer_l2 * self.weights

        #L1 on Biases
        if self.bias_regularizer_l1 > 0:
            dL1 = np.ones_like(self.biases)
            dL1[self.biases < 0] = -1
            self.dbiases += self.bias_regularizer_l1 * dL1
        
        #L2 on biases
        if self.bias_regularizer_l2 > 0:
            self.dbiases += 2 * self.bias_regularizer_l2 * self.biases


        # Gradient on Values / Inputs
        self.dinputs = np.dot(dvalues, self.weights.T)

# Dropout
class Layer_Dropout:
    
    def __init__(self, rate):
        # Invert and store rate to success rate.
        self.rate = 1 - rate
    
    def forward(self, inputs):
        self.inputs = inputs
        self.binary_mask = np.random.binomial(1, self.rate, size=inputs.shape) / \
                           self.rate
        self.output = inputs * self.binary_mask
    
    def backward(self, dvalues):
        self.dinputs = dvalues * self.binary_mask

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
        self.inputs = inputs
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

# Sigmoid Activation
class Activation_Sigmoid:
    def forward(self, inputs):
        self.inputs = inputs
        self.output = 1 / (1 + np.exp(-inputs))
    
    def backward(self, dvalues):
        self.dinputs = dvalues * (1 -self.output) * self.output

# Base/Common Loss Class
class Loss:
    # Calculates the data and regularization losses.
    # given model output and ground truth values.
    def calculate(self, output, y):
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses)
        return data_loss

    def regularization_loss(self, layer):
        # Default to 0
        regularization_loss = 0

        # L1 Regularization - Weights
        # Calculate only when factor greater than 0
        if layer.weight_regularizer_l1 > 0:
            regularization_loss += layer.weight_regularizer_l1 * np.sum(np.abs(layer.weights))

        # L2 Regularization - Weights
        if layer.weight_regularizer_l2 > 0:
            regularization_loss += layer.weight_regularizer_l2 * np.sum(layer.weights * layer.weights)

        # L1 Regularization - Biases
        if layer.bias_regularizer_l1 > 0:
            regularization_loss += layer.bias_regularizer_l1 * np.sum(np.abs(layer.biases))

        # L2 Regularization - Biases
        if layer.bias_regularizer_l2 > 0:
            regularization_loss += layer.bias_regularizer_l2 * np.sum(layer.biases * layer.biases)

        return regularization_loss

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

class Loss_BinaryCrossEntropy(Loss):

    def forward(self, y_pred, y_true):
        # Clip data to prevent division by 0
        # Clip both sides to not drag mean towards any value
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

        # Calculate sample-wise loss
        sample_losses = -(y_true * np.log(y_pred_clipped) + \
                          (1 - y_true) * np.log(1 - y_pred_clipped))
        sample_losses = np.mean(sample_losses, axis=-1)

        return sample_losses

    def backward(self, dvalues, y_true):
        # Number of samples
        samples = len(dvalues)

        # Number of outputs in every sample
        # We'll use the first sample to count them
        outputs = len(dvalues[0])

        # Clip data to prevent division by 0
        # Clip both sides to not drag mean towards any value
        clipped_dvalues = np.clip(dvalues, 1e-7, 1-1e-7)

        # Calculate Gradient
        self.dinputs = -(y_true / clipped_dvalues - \
                         (1 - y_true) / (1 - clipped_dvalues)) / outputs
        
        # Normalize gradient
        self.dinputs = self.dinputs / samples

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

class Optimizer_SGD:
    # Initialize optimizer - set settings,
    # learning rate of 1. Default.
    def __init__(self, learning_rate=1., decay=0., momentum=0.):
        self.learning_rate = learning_rate
        self.decay = decay
        self.current_learning_rate = learning_rate
        self.iterations = 0
        self.momentum = momentum
    
    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * \
                        (1. / (1. + self.decay * self.iterations))

    def update_params(self, layer):
        if self.momentum:
            
            # If layer doesn't contain momentum arrays, create them filled with zeroes.
            if not hasattr(layer, 'weight_momentums'):
                layer.weight_momentums = np.zeros_like(layer.weights)
                layer.bias_momentums = np.zeros_like(layer.biases)
            
            # Build weight updates with momentum.
            # Previous updates multiplied by retain factor
            # And update with current gradients.
            weight_updates = \
                    self.momentum * layer.weight_momentums - \
                    self.current_learning_rate * layer.dweights
            layer.weight_momentums = weight_updates

            # Bias updates
            bias_updates = \
                    self.momentum * layer.bias_momentums - \
                    self.current_learning_rate * layer.dbiases
            layer.bias_momentums = bias_updates
        
        else:
            weight_updates = -self.current_learning_rate * layer.dweights
            bias_updates = -self.current_learning_rate * layer.dbiases

        # Update weights and biases with either vanilla updates or momentum updates.
        layer.weights += weight_updates
        layer.biases += bias_updates

    def post_update_params(self):
        self.iterations += 1

class Optimizer_Adagrad:
    # Initialize optimizer - set settings,
    # learning rate of 1. Default.
    def __init__(self, learning_rate=1., decay=0., epsilon=1e-7):
        self.learning_rate = learning_rate
        self.decay = decay
        self.current_learning_rate = learning_rate
        self.iterations = 0
        self.epsilon = epsilon
    
    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * \
                        (1. / (1. + self.decay * self.iterations))

    def update_params(self, layer):

        # If layer doesn't contain cache arrays, create them filled with zeroes.
        if not hasattr(layer, 'weight_cache'):
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_cache = np.zeros_like(layer.biases)
        
        # Update cache with squared current gradients
        layer.weight_cache += layer.dweights ** 2
        layer.bias_cache += layer.dbiases ** 2

        # Vanilla SGD parameter update + normalization
        # With square rooted cache
        layer.weights += -self.current_learning_rate * \
                         layer.dweights / \
                         (np.sqrt(layer.weight_cache) + self.epsilon)
        layer.biases += -self.current_learning_rate * \
                        layer.dbiases / \
                        (np.sqrt(layer.bias_cache) + self.epsilon)

    def post_update_params(self):
        self.iterations += 1

class Optimizer_RMSprop:

    # Initialize optimizer - set settings,
    def __init__(self, learning_rate=0.001, decay=0., epsilon=1e-7, rho=0.9):
        self.learning_rate = learning_rate
        self.decay = decay
        self.current_learning_rate = learning_rate
        self.iterations = 0
        self.epsilon = epsilon
        self.rho = rho
    
    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * \
                        (1. / (1. + self.decay * self.iterations))

    def update_params(self, layer):

        # If layer doesn't contain cache arrays, create them filled with zeroes.
        if not hasattr(layer, 'weight_cache'):
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_cache = np.zeros_like(layer.biases)
        
        # Update cache with squared current gradients
        layer.weight_cache = self.rho * layer.weight_cache + \
                             (1 - self.rho) * layer.dweights**2
        layer.bias_cache = self.rho * layer.bias_cache + \
                             (1 - self.rho) * layer.dbiases**2

        # Vanilla SGD parameter update + normalization
        # With square rooted cache
        layer.weights += -self.current_learning_rate * \
                         layer.dweights / \
                         (np.sqrt(layer.weight_cache) + self.epsilon)
        layer.biases += -self.current_learning_rate * \
                        layer.dbiases / \
                        (np.sqrt(layer.bias_cache) + self.epsilon)

    def post_update_params(self):
        self.iterations += 1

class Optimizer_Adam:
    # Initialize optimizer - set settings,
    def __init__(self, learning_rate=0.001, decay=0., epsilon=1e-7, beta_1=0.9, beta_2=0.999):
        self.learning_rate = learning_rate
        self.decay = decay
        self.current_learning_rate = learning_rate
        self.iterations = 0
        self.epsilon = epsilon
        self.beta_1 = beta_1
        self.beta_2 = beta_2
    
    # Call once before any parameter updates
    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * \
                                        (1. / (1. + self.decay * self.iterations))

    def update_params(self, layer):
        if not hasattr(layer, 'weight_cache'):
            layer.weight_momentums = np.zeros_like(layer.weights)
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_momentums = np.zeros_like(layer.biases)
            layer.bias_cache = np.zeros_like(layer.biases)
        
        # Update momentum with current gradients
        layer.weight_momentums = self.beta_1 * layer.weight_momentums + \
                                 (1 - self.beta_1) * layer.dweights
        layer.bias_momentums = self.beta_1 * layer.bias_momentums + \
                                 (1 - self.beta_1) * layer.dbiases

        # Get corrected Momentum
        # self.iteration is 0 at first pass
        # and we need to start with 1 here                     
        weight_momentums_corrected = layer.weight_momentums / \
                                    (1 - self.beta_1 ** (self.iterations +1))
        bias_momentums_corrected = layer.bias_momentums / \
                                   (1 - self.beta_1 ** (self.iterations +1))

        # Update cache with square current gradientss
        layer.weight_cache = self.beta_2 * layer.weight_cache + \
                             (1 - self.beta_2) * layer.dweights**2
        layer.bias_cache = self.beta_2 * layer.bias_cache + \
                             (1 - self.beta_2) * layer.dbiases**2

        # Get corrected cache
        weight_cache_corrected = layer.weight_cache / \
                                 (1 - self.beta_2 ** (self.iterations + 1))
        bias_cache_corrected = layer.bias_cache / \
                                 (1 - self.beta_2 ** (self.iterations + 1))

        # Vanilla SGD parameter update + normalization
        # With square rooted cache
        layer.weights += -self.current_learning_rate * \
                         weight_momentums_corrected / \
                         (np.sqrt(weight_cache_corrected) + self.epsilon)
        layer.biases += -self.current_learning_rate * \
                        bias_momentums_corrected / \
                        (np.sqrt(bias_cache_corrected) + self.epsilon)

    def post_update_params(self):
        self.iterations += 1

# Create dataset
# X is coordinates
# y is the class
X, y = spiral_data(samples = 100, classes = 2)

# Reshape labels to be a list of lists
# Inner list contains one output (either 0 or 1)
# per each output neuron, 1 in this case
# Previous: [0 1 1 0 1]
# New: [[0] [1] [1] [0] [1]]
y = y.reshape(-1, 1)

# Create Dense Layer with 2 input features & 64 output values (neurons).
dense1 = Layer_Dense(2, 64, weight_regularizer_l2=5e-4, bias_regularizer_l2=5e-4)

# Create ReLU activation to be used with Dense Layer
activation1 = Activation_ReLU()

# Create Dropout Layer
# dropout1 = Layer_Dropout(0.1)

# Create second Dense layer with 64 input features (output of previous layer)
# and 1 output value
dense2 = Layer_Dense(64, 1)

activation2 = Activation_Sigmoid()

# Create Softmax Classifier's combined loss and activation
# loss_activation = Activation_Softmax_Loss_CategoricalCrossEntropy()

# Create loss function
loss_function = Loss_BinaryCrossEntropy()

# Create Optimizer
# optimizer = Optimizer_SGD(decay=1e-3, momentum=0.85)
optimizer = Optimizer_Adam(decay=5e-7)

# Train in loop
for epoch in range(10001):

    # Perform Forward Pass of the training data through this layer.
    dense1.forward(X)

    # Forward pass through activation func.
    # Takes output of first dense layer.
    activation1.forward(dense1.output)

    # Perform a forward pass through Dropout layer
    # dropout1.forward(activation1.output)

    # Forward pass through second Dense layer.
    # Takes output of activation function from layer 1 (after dropout).
    dense2.forward(activation1.output)
    
    activation2.forward(dense2.output)

    # Perform a Forward pass through the activation/loss function
    # Takes the output of second dense layer here and return loss.
    data_loss = loss_function.calculate(activation2.output, y)
    
    # Calculate Regularization Penalty
    regularization_loss = loss_function.regularization_loss(dense1) + \
                          loss_function.regularization_loss(dense2)

    loss = data_loss + regularization_loss

    # Calculate accuracy from output of activation2 and targets.
    # Calculate values along first axis.
    # Used for probability distributions:
    # predictions = np.argmax(loss_activation.output, axis=1)
    # Binary Classifier validates output of 0 or 1.activation
    predictions = (activation2.output > 0.5) * 1

    # if len(y.shape) == 2:
    #     y = np.argmax(y, axis=1)
    accuracy = np.mean(predictions == y)

    if not epoch % 100:
        print(f'Epoch: {epoch}, ' + 
              f'Acc: {accuracy:.3f}, ' +
              f'Loss: {loss:.3f} (' +
              f'Data Loss: {data_loss:.3f}, ' +
              f'Reg Loss: {regularization_loss:.3f}), ' +
              f'LR: {optimizer.current_learning_rate}, ')

    # Backward Pass - Backpropagation.
    loss_function.backward(activation2.output, y)
    activation2.backward(loss_function.dinputs)
    dense2.backward(activation2.dinputs)
    # dropout1.backward(dense2.dinputs)
    activation1.backward(dense2.dinputs)
    dense1.backward(activation1.dinputs)

    # Apply optimized weights & biases to the 2 layers.
    # The layers store their parameters and gradients (calculated during backpropagation).
    optimizer.pre_update_params()
    optimizer.update_params(dense1)
    optimizer.update_params(dense2)
    optimizer.post_update_params()


# VALIDATE THE MODEL
# Create Test Data

X_test, y_test = spiral_data(samples = 100, classes = 2)
y_test = y_test.reshape(-1, 1)

# Perform a forward pass of our testing data through this layer.
dense1.forward(X_test)

# Perform a forward pass through activation function
# Takes output for first dense layer here.
activation1.forward(dense1.output)

# Perform forward pass through second Dense layer
# Takes outputs of activation function of first layer as inputs
dense2.forward(activation1.output)

activation2.forward(dense2.output)

# Perform forward pass through activation/loss function
# Takes output of second dense layer here and returns loss.
loss = loss_function.calculate(activation2.output, y_test)

# Calculate accuracy from output of activation2 and targets
# Calculate values along first axis
predictions =(activation2.output > 0.5) * 1
# if len(y_test.shape) == 2:
#     y_test = np.argmax(y_test, axis=1)
accuracy = np.mean(predictions == y_test)

print(f'Validation, Acc: {accuracy: .3f}, Loss: {loss:.3f}')
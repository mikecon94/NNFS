# Different versions of Numpy/Python produce different outputs.
# NNFS book uses:
# Python 3.7.5
# Numpy 1.15.0
# Matplotlib 3.1.1

import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import pickle
import copy
import nnfs
from nnfs.datasets import sine_data
from nnfs.datasets import spiral_data
nnfs.init()

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
        self.weights = 0.1 * np.random.randn(n_inputs, n_neurons)
        # Initialize a row vector of zeroes for weights.
        self.biases = np.zeros((1, n_neurons))
        # Self Regularization Strength
        self.weight_regularizer_l1 = weight_regularizer_l1
        self.weight_regularizer_l2 = weight_regularizer_l2
        self.bias_regularizer_l1 = bias_regularizer_l1
        self.bias_regularizer_l2 = bias_regularizer_l2
    
    def forward(self, inputs, training):
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
    
    # Retrieve layer parameters
    def get_parameters(self):
        return self.weights, self.biases
    
    # Set weights & biases in a layer instance
    def set_parameters(self, weights, biases):
        self.weights = weights
        self.biases = biases

# Input Layer - used as a previous layer for the first layer in the model.
class Layer_Input:
    def forward(self, inputs, training):
        self.output = inputs

# Dropout
class Layer_Dropout:
    
    def __init__(self, rate):
        # Invert and store rate to success rate.
        self.rate = 1 - rate
    
    def forward(self, inputs, training):
        self.inputs = inputs
        
        if not training:
            self.output = inputs.copy()
            return

        self.binary_mask = np.random.binomial(1, self.rate, size=inputs.shape) / \
                           self.rate
        self.output = inputs * self.binary_mask
    
    def backward(self, dvalues):
        self.dinputs = dvalues * self.binary_mask

# ReLU Activation
class Activation_ReLU:
    def forward(self, inputs, training):
        self.inputs = inputs
        self.output = np.maximum(0, inputs)

    def backward(self, dvalues):
        # Gradient with respect to inputs on ReLU.
        # Copy next layer and zero out values where the input was less than 0.
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= 0] = 0
    
    def predictions(self, outputs):
        return outputs

class Activation_Softmax:
    def forward(self, inputs, training):
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
    
    # Calculate Predictions for outputs
    def predictions(self, outputs):
        return np.argmax(outputs, axis=1)

# Sigmoid Activation
class Activation_Sigmoid:
    def forward(self, inputs, training):
        self.inputs = inputs
        self.output = 1 / (1 + np.exp(-inputs))
    
    def backward(self, dvalues):
        self.dinputs = dvalues * (1 -self.output) * self.output
    
    def predictions(self, outputs):
        return (outputs > 0.5) * 1

# Inputs passed to output - used for regression model.
class Activation_Linear:
    def forward(self, inputs, training):
        self.inputs = inputs
        self.output = inputs

    def backward(self, dvalues):
        self.dinputs = dvalues.copy()

    def predictions(self, outputs):
        return outputs

# Base/Common Loss Class
class Loss:
    # Set / Remember trainable layers
    def remember_trainable_layers(self, trainable_layers):
        self.trainable_layers = trainable_layers

    # Calculates the data and regularization losses.
    # given model output and ground truth values.
    def calculate(self, output, y, *, include_regularization=False):
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses)
        
        # Add accumulated sum of losses and sample count.
        # This is required for handling batches of data.
        self.accumulated_sum += np.sum(sample_losses)
        self.accumulated_count += len(sample_losses)

        #  If just data loss - return it
        if not include_regularization:
            return data_loss

        # Return the data and regularization losses.
        return data_loss, self.regularization_loss()

    def calculate_accumulated(self, *, include_regularization=False):
        # Calculate mean loss
        data_loss = self.accumulated_sum / self.accumulated_count

        # If just data loss - return it
        if not include_regularization:
            return data_loss
        
        # Return the data and regularization losses
        return data_loss, self.regularization_loss()

    # Reset variables for accumulated loss
    def new_pass(self):
        self.accumulated_sum = 0
        self.accumulated_count = 0

    def regularization_loss(self):
        # Default to 0
        regularization_loss = 0

        # Calculate regularization loss
        # Iterate all trainable layers
        for layer in self.trainable_layers:

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

# Used for Regression Model
# L1 Loss
class Loss_MeanAbsoluteError(Loss):
    def forward(self, y_pred, y_true):
        sample_losses = np.mean(np.abs(y_true - y_pred), axis=-1)
        return sample_losses

    def backward(self, dvalues, y_true):
        # Number of Samples
        samples = len(dvalues)
        # Number of outputs in every sample
        # We'll use the first sample to count them
        outputs = len(dvalues[0])
        
        # Calculate Gradient
        self.dinputs = np.sign(y_true - dvalues) / outputs
        # Normalize gradient
        self.dinputs = self.dinputs / samples

# Used for Regression Model.
# Squares the error (higher penalty the further away the value is)
# L2 Loss
class Loss_MeanSquaredError(Loss):
    def forward(self, y_pred, y_true):
        #Calculate Loss
        # Calculate mean across outputs, for each sample separately.
        sample_losses = np.mean((y_true - y_pred)**2, axis=-1)
        return sample_losses

    def backward(self, dvalues, y_true):
        # Number Of Samples
        samples = len(dvalues)
        # Number of outputs in every sample
        # We'll use the first sample to count them
        outputs = len(dvalues[0])

        # Gradient on values
        self.dinputs = -2 * (y_true - dvalues) / outputs
        # Normalise gradient
        self.dinputs = self.dinputs / samples

# Softmax Classifier - Combined Softmax activation
# with cross-entropy loss for faster backward step.
class Activation_Softmax_Loss_CategoricalCrossEntropy():
    
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

class Accuracy:
    # Calculates an accuracy
    # Given predictions and ground truth values
    def calculate(self, predictions, y):
        # Get comparison results
        comparisons = self.compare(predictions, y)
        # Calculate an accuracy
        accuracy = np.mean(comparisons)
        
        # Add accumulated sum of matching values and sample count
        self.accumulated_sum += np.sum(comparisons)
        self.accumulated_count += len(comparisons)
        
        # Return Accuracy
        return accuracy
    
    # Calculate Accumulated Accuracy
    def calculate_accumulated(self):
        # calculate an accuracy
        accuracy = self.accumulated_sum /self.accumulated_count

        # return the data and regularization loss
        return accuracy

    # Resets accumulated values. Called at the start of each epoch.
    def new_pass(self):
        self.accumulated_sum = 0
        self.accumulated_count = 0

class Accuracy_Regression(Accuracy):
    def __init__(self):
        self.precision = None
    
    # Calculates precision value
    # based on passed in ground truth
    def init(self, y, reinit=False):
        if self.precision is None or reinit:
            self.precision = np.std(y) / 250

    def compare(self, predictions, y):
        return np.absolute(predictions - y) < self.precision

class Accuracy_Categorical(Accuracy):
    def __init__(self, *, binary=False):
        # Binary mode?
        self.binary = binary
    
    # No initialization is needed
    def init(self, y):
        pass

    # Compares predictions to ground truth values
    def compare(self, predictions, y):
        if not self.binary and len(y.shape) == 2:
            y = np.argmax(y, axis=1)
        return predictions == y

class Model:
    def __init__(self):
        # Create a list of network objects
        self.layers = []
        self.softmax_classifier_output = None

    # Add objects to the model
    def add(self, layer):
        self.layers.append(layer)

    # Set Loss & Optimizer
    # * parameter definitions notes that the subsequent parameters are keyword arguments.
    # No value is defined so they are required keyword arguments.
    def set(self, *, loss=None, optimizer=None, accuracy=None):
        if loss is not None:
            self.loss = loss
        if optimizer is not None:
            self.optimizer = optimizer
        if accuracy is not None:
           self.accuracy = accuracy

    def finalize(self):
        # Create and set the input layer
        self.input_layer = Layer_Input()

        # Count all the objects
        layer_count = len(self.layers)

        # Initialize list of trainable layers
        self.trainable_layers = []

        # Iterate the objects
        for i in range(layer_count):
            # If it's the first layer,
            # the previous layer object is the input layer
            if i == 0:
                self.layers[i].prev = self.input_layer
                self.layers[i].next = self.layers[i+1]
            elif i < layer_count - 1:
                self.layers[i].prev = self.layers[i-1]
                self.layers[i].next = self.layers[i+1]
            else:
                self.layers[i].prev = self.layers[i-1]
                self.layers[i].next = self.loss
                # Save model's output
                self.output_layer_activation = self.layers[i]

            # If layer contains an attribute called "weights",
            # it's a trainable layer - 
            # Add it to hte list of trainable layers
            # We don't need to check for biases -
            # Checking for weights is enough
            if hasattr(self.layers[i], 'weights'):
                self.trainable_layers.append(self.layers[i])

            # If output activation is Softmax and
            # Loss function is Categorical Cross-Entropy
            # create an object of combined activation
            # and loss function containing
            # faster gradient calculation
            if isinstance(self.layers[-1], Activation_Softmax) and isinstance(self.loss, Loss_CategoricalCrossEntropy):
                # Create an object of combined activation and loss functions
                self.softmax_classifier_output = \
                    Activation_Softmax_Loss_CategoricalCrossEntropy()
        
        # Update loss object with trainable layers
        # Only train layers if we parameters haven't be passed in initially.
        if self.loss is not None:
            self.loss.remember_trainable_layers(
                self.trainable_layers
            )
    
    def evaluate(self, X_val, y_val, *, batch_size=None):
        # Default value if batch_size isn't set.
        validation_steps = 1

        # Calculate number of steps
        if batch_size is not None:
            validation_steps = len(X_val) // batch_size
            if validation_steps * batch_size < len(X_val):
                validation_steps += 1

        # Reset accumulated values in loss and accuracy objects
        self.loss.new_pass()
        self.accuracy.new_pass()
        
        # Iterate over steps
        for step in range(validation_steps):
            # If batch size is not set - train using one step and full dataset.
            if batch_size is None:
                batch_X = X_val
                batch_y = y_val
            else:
                batch_X = X_val[
                    step*batch_size:(step+1)*batch_size
                ]
                batch_y = y_val[
                    step*batch_size:(step+1)*batch_size
                ]
            output = self.forward(batch_X, training=False)
            self.loss.calculate(output, batch_y)
            
            # Get predictions and calculate accuracy
            predictions = self.output_layer_activation.predictions(output)
            self.accuracy.calculate(predictions, batch_y)

        validation_loss = self.loss.calculate_accumulated()
        validation_accuracy = self.accuracy.calculate_accumulated()

        print(f'validation, ' +
              f'acc: {validation_accuracy:.3f}, ' +
              f'loss: {validation_loss:.3f}')

    def train(self, X, y, *, epochs=1, batch_size = None, print_every=1, validation_data=None):
        self.accuracy.init(y)

        # Default value if batch size is not set.
        train_steps = 1

        # If there is validation data passed,
        # Set default number of steps for validation as well
        if validation_data is not None:
            validation_steps = 1
            # for better readability
            X_val, y_val = validation_data

        # Calculate number of steps
        if batch_size is not None:
            train_steps = len(X) // batch_size

            # Dividing rounds down. If there are remaining data it will be missed.
            # Add an extra step to cover all samples.
            if train_steps * batch_size < len(X):
                train_steps += 1

            if validation_data is not None:
                validation_steps = len(X_val) // batch_size
                if validation_steps * batch_size < len(X_val):
                    validation_steps += 1
            
        for epoch in range(1, epochs+1):

            print(f'epoch: {epoch}')          
            # Reset accumulated values in loss and accuracy
            self.loss.new_pass()
            self.accuracy.new_pass()
            for step in range(train_steps):
            
                # If batch size is not set - train using on step and full dataset.
                if batch_size is None:
                    batch_X = X
                    batch_y = y
                # Otherwise slice a batch
                else:
                    batch_X = X[step*batch_size:(step+1)*batch_size]
                    batch_y = y[step*batch_size:(step+1)*batch_size]

                # Perform forward pass
                output = self.forward(batch_X, training=True)

                # Calculate Loss
                data_loss, regularization_loss = self.loss.calculate(output, batch_y, include_regularization=True)
                loss = data_loss + regularization_loss

                # Get predictions & calculate an accuracy
                predictions = self.output_layer_activation.predictions(output)
                accuracy = self.accuracy.calculate(predictions, batch_y)

                self.backward(output, batch_y)
                
                # Optimize (update parameters)
                self.optimizer.pre_update_params()
                for layer in self.trainable_layers:
                    self.optimizer.update_params(layer)
                self.optimizer.post_update_params()

                # Print summary
                if not step % print_every or step == train_steps - 1:
                    print(f'step: {step}, ' +
                        f'acc: {accuracy:.3f}, ' +
                        f'loss: {loss:.3f} (' +
                        f'data_loss: {data_loss:.3f}, ' +
                        f'reg_loss: {regularization_loss:.3f}), ' +
                        f'lr: {self.optimizer.current_learning_rate}')
            
            # Print epoch loss and accuracy
            epoch_data_loss, epoch_regularization_loss = \
                self.loss.calculate_accumulated(include_regularization=True)
            epoch_loss = epoch_data_loss + epoch_regularization_loss
            epoch_accuracy = self.accuracy.calculate_accumulated()
            print(f'training, ' +
                  f'acc: {epoch_accuracy:.3f}, ' +
                  f'loss: {epoch_loss:.3f}, ' +
                  f'data_loss: {epoch_data_loss:.3f}, ' +
                  f'reg_loss: {epoch_regularization_loss:.3f}, ' +
                  f'lr: {self.optimizer.current_learning_rate:.3f}')

            if validation_data is not None:
                # Evaluate the model:
                # Look at examples/starred-expressions.py for *validation_data
                self.evaluate(*validation_data, batch_size=batch_size)

    def forward(self, X, training):
        # Call forward method on the input layer
        # This will set the output property that
        # the first layer in "prev" object is expecting
        self.input_layer.forward(X, training)

        # Call forward method of every object in a chain
        # Pass output of the previous object as a parameter
        for layer in self.layers:
            layer.forward(layer.prev.output, training)
        
        # "layer" is now the last object from hte list
        # Return its output
        return layer.output

    def backward(self, output, y):
        if self.softmax_classifier_output is not None:
            # First call backward method
            # on the combined activation/loss
            # this will set dinputs property
            self.softmax_classifier_output.backward(output, y)

            # Since we'll not call backward method of the last layer
            # Which is Softmax Activation
            # as we used combined activation/loss
            # object, let's set dinputs in this object
            self.layers[-1].dinputs = \
                self.softmax_classifier_output.dinputs

            # Call backward method going through
            # all the objects but last
            # in reversed order passing dinputs as a parameter
            for layer in reversed(self.layers[:-1]):
                layer.backward(layer.next.dinputs)
            return

        # First call backward method on the loss
        # this will set dinputs property that the last
        # layer will try to access shortly
        self.loss.backward(output, y)

        # Call backward method going through all the objects
        # in reversed order passing dinputs as a parameter
        for layer in reversed(self.layers):
            layer.backward(layer.next.dinputs)

    # Retrieves and returns parameters of trainable layers
    def get_parameters(self):
        # create a list of params
        parameters = []

        # Iterate over trainable layers and get their params.
        for layer in self.trainable_layers:
            parameters.append(layer.get_parameters())
        
        # return the list
        return parameters

    # Update the model with new parameters
    def set_parameters(self, parameters):
        # Iterate over the parameters and layers
        # and update each layers with each set of the parameters
        # zip() takes a list of parameters and a list of layers and returns an iterator containing
        # tuples of 0th elements of both lists, then the 1st elements of both lists, the 2nd elements from both
        # lists, and so on
        for parameter_set, layer in zip(parameters, self.trainable_layers):
            layer.set_parameters(*parameter_set)

    # Predicts on the samples
    # Can still predict in batches.
    # X should be shape (1, 784)
    # Where 1 is number of samples and 784 is the number of features (image pixels)
    def predict(self, X, *, batch_size=None):
        # Default value if batch size is not being set
        prediction_steps = 1
        # Calculate the number of steps
        if batch_size is not None:
            prediction_steps = len(X) // batch_size
            if prediction_steps * batch_size < len(X):
                prediction_steps += 1            
        # Model Ouputs
        output = []
        # Iterate over the steps / batches
        for step in range(prediction_steps):
            if batch_size is None:
                batch_X = X
            else:
                batch_X = X[step*batch_size:(step+1)*batch_size]
            
            # Perform the forward pass
            batch_output = self.forward(batch_X, training=False)
            # Append batch prediction to list of predictions
            output.append(batch_output)
        return np.vstack(output)

    # Save parameters to a file
    def save_parameters(self, path):
        # Open a file in the bianry-write mode
        # and save parameters to it
        with open(path, 'wb') as f:
            pickle.dump(self.get_parameters(), f)
        
    # Load parameters from a file
    def load_parameters(self, path):
        # Open a file in birary read mode
        with open(path, 'rb') as f:
            self.set_parameters(pickle.load(f))

    # Saves the model
    def save(self, path):
        # Make a deep copy of current model instance
        model = copy.deepcopy(self)
        # Reset accumulated values in loss and accuracy objects
        model.loss.new_pass()
        model.accuracy.new_pass()
        # Remove data from input layer and gradients from loss object
        model.input_layer.__dict__.pop('output', None)
        model.loss.__dict__.pop('dinputs', None)

        # For each layer remove inputs, output and dinputs properties
        for layer in model.layers:
            for property in ['inputs', 'output', 'dinputs', 'dweights', 'dbiases']:
                layer.__dict__.pop(property, None)
        
        # Open a file in binary write mode and save the model
        with open(path, 'wb') as f:
            pickle.dump(model, f)
    
    # Loads and returns a model
    @staticmethod
    def load(path):
        # Open file in binary-read mode, load a model
        with open(path, 'rb') as f:
            model = pickle.load(f)
        # Return a model
        return model

# Create dataset
# X, y = spiral_data(samples=1000, classes=3)
# X_test, y_test = spiral_data(samples=100, classes=3)
def load_mnist_dataset(dataset, path):
    labels = os.listdir(os.path.join(path, dataset))
    X = []
    y = []
    for label in labels:
        for file in os.listdir(os.path.join(path, dataset, label)):
            image = cv2.imread(os.path.join(path, dataset, label, file), cv2.IMREAD_UNCHANGED)
            X.append(image)
            y.append(label)
    return np.array(X), np.array(y).astype('uint8')

def create_data_mnist(path):
    X, y = load_mnist_dataset('train', path)
    X_test, y_test = load_mnist_dataset('test', path)
    return X, y, X_test, y_test

X, y, X_test, y_test = create_data_mnist('fashion_mnist_images')

# Shuffle the training dataset
keys = np.array(range(X.shape[0]))
np.random.shuffle(keys)
X = X[keys]
y = y[keys]

# Scale and reshape samples
X = (X.reshape(X.shape[0], -1).astype(np.float32) - 127.5) / 127.5
X_test = (X_test.reshape(X_test.shape[0], -1).astype(np.float32) - 127.5) / 127.5

def train_new_model(path):
    # Instantiate the model
    model = Model()

    # Add Layers
    model.add(Layer_Dense(X.shape[1], 128))
    model.add(Activation_ReLU())
    model.add(Layer_Dense(128, 128))
    model.add(Activation_ReLU())
    model.add(Layer_Dense(128, 10))
    model.add(Activation_Softmax())
    model.set(
        loss = Loss_CategoricalCrossEntropy(),
        optimizer=Optimizer_Adam(decay=1e-3),
        accuracy=Accuracy_Categorical()
    )
    model.finalize()
    model.train(X, y, validation_data=(X_test, y_test), epochs=10, batch_size=128, print_every=100)

    # Retrieve parameters
    # parameters = model.get_parameters()
    # Save model parameters
    model.save_parameters(path)

# Create new model and load it with saved weights & biases
def create_and_load_model(path):
    # New Model
    model = Model()
    model.add(Layer_Dense(X.shape[1], 128))
    model.add(Activation_ReLU())
    model.add(Layer_Dense(128, 128))
    model.add(Activation_ReLU())
    model.add(Layer_Dense(128, 10))
    model.add(Activation_Softmax())

    # Set loss and accuracy objects
    # We don't set optimizer object this time - no need to do it
    # as we aren't training the model.
    model.set(loss=Loss_CategoricalCrossEntropy(), accuracy=Accuracy_Categorical())

    model.finalize()

    # Load the parameters from the file from previously trained model
    model.load_parameters(path)
    model.evaluate(X_test, y_test)
    # Save the full model
    model.save('fashion_mnist.model')

model = Model.load('fashion_mnist.model')

# Evaluate the model with test data
# model.evaluate(X_test, y_test)

# Predict on first 5 samples from validation dataset
# and print the result
confidences = model.predict(X_test[:5])
print(confidences)
predictions = model.output_layer_activation.predictions(confidences)
print(predictions)
print(y_test[:5])

fashion_mnist_labels = {
    0: 'T-shirt/top',
    1: 'Trousers',
    2: 'Pullover',
    3: 'Dress',
    4: 'Coat',
    5: 'Sandal',
    6: 'Shirt',
    7: 'Sneaker',
    8: 'Bag',
    9: 'Ankle Boot'
}

for prediction in predictions:
    print(fashion_mnist_labels[prediction])


# Predict with images outside of the MNIST dataset
image_data = cv2.imread('tshirt.png', cv2.IMREAD_GRAYSCALE)
image_data = cv2.resize(image_data, (28, 28))
image_data = 255 - image_data
image_data = (image_data.reshape(1, -1).astype(np.float32) - 127.5) / 127.5
confidences = model.predict(image_data)
predictions = model.output_layer_activation.predictions(confidences)
prediction = fashion_mnist_labels[predictions[0]]
print("T-Shirt Input: " + prediction)

image_data = cv2.imread('pants.png', cv2.IMREAD_GRAYSCALE)
image_data = cv2.resize(image_data, (28, 28))
image_data = 255 - image_data
image_data = (image_data.reshape(1, -1).astype(np.float32) - 127.5) / 127.5
confidences = model.predict(image_data)
predictions = model.output_layer_activation.predictions(confidences)
prediction = fashion_mnist_labels[predictions[0]]
print("Trousers Input: " + prediction)
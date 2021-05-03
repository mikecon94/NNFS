import matplotlib.pyplot as plt
import numpy as np
import nnfs
from nnfs.datasets import sine_data

nnfs.init()

class Activation_Linear:
    def forward(self, inputs):
        self.inputs = inputs
        self.output = inputs

    def backward(self, dvalues):
        self.dinputs = dvalues.copy()

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

X, y = sine_data()
plt.plot(X, y)
plt.show()
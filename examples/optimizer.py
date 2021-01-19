import matplotlib.pyplot as plt
import nnfs
from nnfs.datasets import vertical_data
from nnfs.datasets import spiral_data
import numpy as np
import sys, os
sys.path.insert(1, os.path.join(sys.path[0], '..'))
import main

nnfs.init()
# X, y = vertical_data(samples=100, classes=3)
X, y = spiral_data(samples=100, classes=3)
# plt.scatter(X[:, 0], X[:, 1], c=y, s=1, cmap='brg')
# plt.show()

dense1 = main.Layer_Dense(2, 3)
activation1 = main.Activation_ReLU()
dense2 = main.Layer_Dense(3, 3)
activation2 = main.Activation_Softmax()

loss_function = main.Loss_CategoricalCrossEntropy()

lowest_loss = 9999999 # some initial value
best_dense1_weights = dense1.weights.copy()
best_dense1_biases = dense1.biases.copy()
best_dense2_weights = dense2.weights.copy()
best_dense2_biases = dense2.biases.copy()

# Iterate 10000 times using random values for weights and biases.
# Save the weights and biases if they generate the lowest-seen loss.
for iteration in range(100000):
    dense1.weights += 0.05 * np.random.randn(2, 3)
    dense1.biases += 0.05 * np.random.randn(1, 3)
    dense2.weights += 0.05 * np.random.randn(3, 3)
    dense2.biases += 0.05 * np.random.randn(1, 3)

    dense1.forward(X)
    activation1.forward(dense1.output)
    dense2.forward(activation1.output)
    activation2.forward(dense2.output)

    loss = loss_function.calculate(activation2.output, y)

    predictions = np.argmax(activation2.output, axis=1)
    accuracy = np.mean(predictions == y)

    if loss < lowest_loss:
        print("New set of weights found, iteration:", iteration,
              "Loss:", loss,
              "Accuracy:", accuracy)
        best_dense1_weights = dense1.weights.copy()
        best_dense1_biases = dense1.biases.copy()
        best_dense2_weights = dense2.weights.copy()
        best_dense2_biases = dense2.biases.copy()
        lowest_loss = loss
    else:
        dense1.weights = best_dense1_weights.copy()
        dense1.biases = best_dense1_biases.copy()
        dense2.weights = best_dense2_weights.copy()
        dense2.biases = best_dense2_biases.copy()

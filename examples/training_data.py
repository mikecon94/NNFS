import matplotlib.pyplot as plt
import nnfs
import numpy as np
from nnfs.datasets import spiral_data

# Sets the random seed to 0
# Creates a float32 dtype default
# Overrides original dot product from NumPy
nnfs.init()

X, y = spiral_data(samples=100, classes=3)
print(X, y)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='brg')
# plt.show()
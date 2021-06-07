import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import nnfs

nnfs.init()

# Loads a MNIST dataset
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

# Load MNIST Train & Test set
def create_data_mnist(path):
    X, y = load_mnist_dataset('train', path)
    X_test, y_test = load_mnist_dataset('test', path)
    return X, y, X_test, y_test

X, y, X_test, y_test = create_data_mnist('fashion_mnist_images')

# Scale the data down to -1 to 1.
X = (X.astype(np.float32) - 127.5) / 127.5
X_test = (X_test.astype(np.float32) - 127.5) / 127.5
print(X.min(), X.max())
print(X.shape)

# Flatten the data for the NN
example = np.array([[1, 2], [3, 4]])
flattened = example.reshape(-1)
print(example)
print(example.shape)
print(flattened)
print(flattened.shape)

# Flattens to (60,000, 784)
# 28x28 = 784
X = X.reshape(X.shape[0], -1)
X_test = X_test.reshape(X_test.shape[0], -1)

# Shuffle the dataset
keys = np.array(range(X.shape[0]))
print(keys[:10])
np.random.shuffle(keys)
print(keys[:10])

X = X[keys]
y = y[keys]
print(y[:15])

plt.imshow((X[8].reshape(28, 28)))
plt.show()
print(y[8])
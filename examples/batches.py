

batch_size = 2
X = [1, 2, 3, 4]

# Integer division is used as steps must be whole numbers.
print(len(X)//batch_size)

# If there are remaining items in the batch we add 1 to steps to get them all.
X = [1, 2, 3, 4, 5]
steps = len(X) // batch_size
if steps * batch_size < len(X):
    steps += 1
print(steps)

import nnfs
from nnfs.datasets import spiral_data

nnfs.init()

# Create Dataset
X, y = spiral_data(samples = 100, classes = 3)

EPOCHS = 10
BATCH_SIZE = 128

# Calculate the number of steps
steps = X.shape[0] // BATCH_SIZE
if steps * BATCH_SIZE < X.shape[0]:
    steps += 1

for epoch in range(EPOCHS):
    for step in range(steps):
        # During each step in each epoch, we select a slice of the training data.
        batch_X = X[step*BATCH_SIZE:(step+1)*BATCH_SIZE]
        batch_y = y[step*BATCH_SIZE:(step+1)*BATCH_SIZE]
        # Now perform forward pass, loss calculation
        # backward pass and update parameters.
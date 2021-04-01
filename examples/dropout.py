import random

dropout_rate = 0.5

example_output = [0.27, -1.03, 0.67, 0.99, 0.05,
                  -0.37, -2.01, 1.13, -0.07, 0.73]

while True:
    # Randomly choose index and set value to 0
    index = random.randint(0, len(example_output) - 1)
    example_output[index] = 0

    # We might set an index that already is zeroed
    # There are different ways of overcoming this problem,
    # For simpliciy we count values that are exactly 0
    # while it's extremely rare in real model that weights
    # are exactly 0, this is not the best method for sure.
    dropped_out = 0
    
    for value in example_output:
        if value == 0:
            dropped_out += 1
    
    # If required number of outputs is zeroed - leave the loop
    if dropped_out / len(example_output) >= dropout_rate:
        break

print(example_output)


# Consider Bernoulli distribution a special case of a Binomial distribution
# with n=1
# numpy.random.binomial
# n is the number of concurrent experiments and returns number of successes
# from these n experiments.

import numpy as np
# np.random.binomial(n, p, size)
# Coin Toss
# n is how many tosses of the coin
# p is probability for toss results to be a 1
# Result is sum of all toss results
# size is how many "tests" to run.
# Return is list of results.
print(np.random.binomial(2, 0.5, size=10))

# Dropout Layer
# We want a filter where intended dropout % is represented as 0.
# Everything else as 1.
dropout_rate = 0.20
print(np.random.binomial(1, 1-dropout_rate, size=5))

# NN Layer's Output
example_output = np.array([0.27, -1.03, 0.67, 0.99, 0.05,
                           -0.37, -2.01, 1.13, -0.07, 0.73])
dropout_rate = 0.3

example_output *= np.random.binomial(1, 1-dropout_rate, example_output.shape)
print(example_output)


# To ensure magnitued of inputs to layers remain the same
# Regardless of whether dropout is used
# the values are scaled up to match the dropout rate during training
# so the sums of inputs are equal during prediction.
dropout_rate = 0.2
example_output = np.array([0.27, -1.03, 0.67, 0.99, 0.05,
                           -0.37, -2.01, 1.13, -0.07, 0.73])
print(f'sum initial: {sum(example_output)}')
sums = []

for i in range(10000):
    example_output2 = example_output * \
                    np.random.binomial(1, 1 - dropout_rate, example_output.shape) / \
                    (1 - dropout_rate)
    sums.append(sum(example_output2))
print(f'mean sum: {np.mean(sums)}')

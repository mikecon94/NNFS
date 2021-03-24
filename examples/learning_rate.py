starting_learning_rate = 1.
learning_rate_decay = 0.1 # This is an agressive decay rate.

for step in range(20):
    learning_rate = starting_learning_rate * \
                    (1. / (1 + learning_rate_decay * step))
    print("Step: ", step, "Learning Rate:", learning_rate)
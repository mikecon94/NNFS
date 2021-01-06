import math

# Example output from output layer of NN
softmax_output = [0.7, 0.1, 0.2]

# Ground Truth
target_output = [1, 0, 0]

# loss = -(math.log(softmax_output[0])*target_output[0] +
#         math.log(softmax_output[1])*target_output[1] +
#         math.log(softmax_output[2])*target_output[2])

# For categorical cross-entropy
# 1 class will be set to 1 and the rest 0.
# The 0 classes will multiply to 0 so not required.
# This means we just need:
loss = -math.log(softmax_output[0])

print(loss)
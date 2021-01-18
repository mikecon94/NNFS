import math
import numpy as np

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

# Probabilities for 3 samples.
# Classes = Dog, Cat, Human
softmax_outputs = np.array([[0.7, 0.1, 0.2],
                            [0.1, 0.5, 0.4],
                            [0.02, 0.9, 0.08]])
class_targets = [0, 1, 1] #Dog, Cat, Cat

print("ZIP Function:")
for targ_idx, distribution in zip(class_targets, softmax_outputs):
    print(distribution[targ_idx])

# NP let's us simplify this:
print("NumPy:")
print(softmax_outputs[[0, 1, 2], class_targets])

# We can then use range to go through full batch without hard-coding indices.
print("NumPy - Dynamic Batch Size:")
print(softmax_outputs[
    range(len(softmax_outputs)), class_targets
])

# Now Apply negative log to each item:
print("With Negative Log:")
print(-np.log(softmax_outputs[
    range(len(softmax_outputs)), class_targets
]))

# Calculate Average Loss
neg_log = -np.log(softmax_outputs[
            range(len(softmax_outputs)), class_targets
          ])
average_loss = np.mean(neg_log)
print("Average Loss:")
print(average_loss)


softmax_outputs = np.array([[0.7, 0.1, 0.2],
                            [0.1, 0.5, 0.4],
                            [0.02, 0.9, 0.08]])
class_targets = np.array([[1, 0, 0],
                          [0, 1, 0],
                          [0, 1, 0]])
# Probabilities for target values
# Categorical labels
if len(class_targets.shape) == 1:
    print("Sparse Targets")
    correct_confidences = softmax_outputs[
                            range(len(softmax_outputs)),
                            class_targets
                          ]
elif len(class_targets.shape) == 2:
    print("One Hot Targets", np.sum(softmax_outputs*class_targets, axis=1))
    correct_confidences = np.sum(
                            softmax_outputs*class_targets,
                            axis=1
                          )
neg_log = -np.log(correct_confidences)
average_loss=np.mean(neg_log)
print(average_loss)



y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)
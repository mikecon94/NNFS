import matplotlib.pyplot as plt
import numpy as np

def f(x):
    # return 2*x
    return 2*x**2

x = np.array(range(5))
y = f(x)
print(x)
print(y)
# plt.plot(x, y)
# plt.show()

# With a linear function any 2 points on the line will give the same value
# With a non-linear function we will get various slopes.
print((y[1]-y[0]) / x[1] - x[0])

# We can approximate a slope at a point x by adding a very small delta.
p2_delta = 0.0001
x1 = 1
x2 = x1 + p2_delta
y1 = f(x1) # Derivative point
y2 = f(x2) # Other very close point
approximate_derivative = (y2-y1)/(x2-x1)
print(approximate_derivative)

# np.arange(start, stop, step) to give us smoother line.
x = np.arange(0, 5, 0.001)
y = f(x)
# plt.plot(x, y)
p2_delta = 0.0001
x1 = 2
x2 = x1 + p2_delta
y1 = f(x1)
y2 = f(x2)
print((x1, y1), (x2, y2))

# Derivative approximation and y-intercept for tangent line
approximate_derivative = (y2-y1) / (x2-x1)
b = y2 - approximate_derivative * x2
# Approximate derivative & b are constant for this function.
def tangent_line(x):
    return approximate_derivative*x + b

# Plot the tangent line
# +/- 0.9 to draw the tangent line on the graph
# Then calculate the y for given x using the tangent line function
to_plot = [x1-0.9, x1, x1+0.9]
# plt.plot(to_plot, [tangent_line(i) for i in to_plot])
print('Approximate derivative for f(x)', f'where x = {x1} is {approximate_derivative}')

x = np.array(np.arange(0, 5, 0.001))
y = f(x)

plt.plot(x, y)
colors = ['k', 'g', 'r', 'b', 'c']

def approximate_tangent_line(x, approximate_derivative):
    return (approximate_derivative * x) + b

for i in range(5):
    p2_delta = 0.0001
    x1 = i
    x2 = x1 + p2_delta
    y1 = f(x1)
    y2 = f(x2)
    print((x1, y1), (x2, y2))

    approximate_derivative = (y2 - y1) / (x2 - x1)
    b = y2 - (approximate_derivative * x2)

    to_plot = [x1 - 0.9, x1, x1 + 0.9]

    plt.scatter(x1, y1, c=colors[i])
    plt.plot([point for point in to_plot],
              [approximate_tangent_line(point, approximate_derivative)
                for point in to_plot],
              c=colors[i])
    print('Approximate derivative for f(x)', f'where x = {x1} is {approximate_derivative}')

plt.show()
import numpy as np

# Inputs
inputs = [1.0, 2.0, 3.0, 2.5]

# Weights for 3 neurons
weights = [
    [0.2, 0.8, -0.5, 1.0],
    [1.0, -0.91, 0.26, -0.5],
    [-0.26, -0.27, 0.17, 0.87]
]

# Bias for each neuron
biases = [2.0, 3.0, 0.5]

# Calculate the outputs of all 3 neurons
layer_outputs = np.dot(weights, inputs) + biases

print(layer_outputs)

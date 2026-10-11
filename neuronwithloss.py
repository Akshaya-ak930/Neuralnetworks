import numpy as np
import nnfs
from nnfs.datasets import spiral_data

nnfs.init()

# Dense layer
class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases


# ReLU activation
class Activation_ReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)


# Softmax activation
class Activation_Softmax:
    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities


# Common loss class
class Loss:
    def calculate(self, output, y):
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses)
        return data_loss


# Categorical Cross-Entropy Loss
class Loss_CategoricalCrossentropy(Loss):
    def forward(self, y_pred, y_true):

        samples = len(y_pred)

        # Clip values to avoid log(0)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

        # If labels are integers
        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[
                range(samples), y_true
            ]

        # If labels are one-hot encoded
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(
                y_pred_clipped * y_true,
                axis=1
            )

        negative_log_likelihoods = -np.log(correct_confidences)

        return negative_log_likelihoods


# Create dataset
X, y = spiral_data(samples=100, classes=3)

# Create first Dense layer
dense1 = Layer_Dense(2, 3)

# Create ReLU activation
activation1 = Activation_ReLU()

# Create second Dense layer
dense2 = Layer_Dense(3, 3)

# Create Softmax activation
activation2 = Activation_Softmax()

# Create loss function
loss_function = Loss_CategoricalCrossentropy()


# Forward pass through first Dense layer
dense1.forward(X)

# Forward pass through ReLU
activation1.forward(dense1.output)

# Forward pass through second Dense layer
dense2.forward(activation1.output)

# Forward pass through Softmax
activation2.forward(dense2.output)


# Print first 5 Softmax outputs
print("Softmax Outputs:")
print(activation2.output[:5])


# Calculate loss
loss = loss_function.calculate(activation2.output, y)

print("Loss:", loss)


# Calculate accuracy
predictions = np.argmax(activation2.output, axis=1)

# If labels are one-hot encoded
if len(y.shape) == 2:
    y = np.argmax(y, axis=1)

accuracy = np.mean(predictions == y)

print("Accuracy:", accuracy)

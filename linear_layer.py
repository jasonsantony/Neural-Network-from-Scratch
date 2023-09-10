import numpy as np

from layer import *


class LinearLayer(Layer):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.weights = np.random.randn(input_size, output_size)
        self.biases = np.zeros((1, output_size))

    # Compute Z = XW + b
    def step_forward(self, inputs):
        self.inputs = inputs
        self.outputs = self.inputs @ self.weights + self.biases
        return self.outputs

    # Compute d_Z/d_X
    def step_backward(self, d_outputs, learning_rate):
        d_inputs = d_outputs @ self.weights.T
        d_weights = self.inputs.T @ d_outputs
        d_biases = d_outputs

        # update weights and biases
        self.weights -= learning_rate * d_weights
        self.biases -= learning_rate * d_biases

        return d_inputs

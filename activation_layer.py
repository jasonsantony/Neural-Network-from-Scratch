from layer import *


class ActivationLayer(Layer):
    def __init__(self, activation_function):
        super().__init__()
        self.activation_function = activation_function

    # Compute Y_hat = A(Z)
    def step_forward(self, inputs):
        self.inputs = inputs
        self.outputs = self.activation_function(self.inputs)
        return self.outputs

    # Compute d_Y_hat/d_X = d_Y_hat/d_Z ⊙ A'(Z)
    def step_backward(self, d_outputs, learning_rate):
        return d_outputs * self.activation_function(self.inputs, prime=True)

class Layer:
    def __init__(self):
        self.inputs = None
        self.outputs = None

    # One step of forward propagation
    def step_forward(self, inputs):
        raise NotImplementedError

    # One step of backward propagation
    def step_backward(self, d_outputs, learning_rate):
        raise NotImplementedError

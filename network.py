import numpy as np


class Network:
    def __init__(self):
        self.layers = []
        self.loss_function = None

    # Set the loss function of the network
    def set_loss_function(self, loss_function):
        self.loss_function = loss_function

    # Add a layer to the network
    def add_layer(self, layer):
        self.layers.append(layer)

    # Make predictions with current network parameters
    # `loss` and `one_hot_acc` are basically verbose flags
    # `one_hot_acc` meant for use with one-hot predictions
    def predict(
        self,
        X_test: np.ndarray,
        Y_test: np.ndarray = None,
        loss: bool = False,
        one_hot_acc: bool = False,
    ):
        samples = len(X_test)
        prediction = []

        for s in range(samples):
            outputs = X_test[s]
            for layer in self.layers:
                outputs = layer.step_forward(outputs)
            prediction.append(outputs)

        if loss:
            error = 0
            for s in range(samples):
                error += self.loss_function(Y_test[s], prediction[s])

            error /= samples
            print(f"Error: {error}")

        if one_hot_acc:
            accuracy = 0
            for s in range(samples):
                if np.argmax(Y_test[s]) == np.argmax(prediction[s]):
                    accuracy += 1

            accuracy /= samples
            print(f"Accuracy: {accuracy}")

        return prediction

    # Train parameters of the network
    def train(
        self,
        X_train: np.ndarray,
        Y_train: np.ndarray,
        learning_rate: float,
        epochs: int,
        mb_size: int,
        verbose: bool = False,
    ):
        samples = len(X_train)

        if mb_size < 1:
            mb_size = samples
        elif mb_size > samples:
            mb_size = samples

        for e in range(epochs):
            error = 0
            for mb in range(samples // mb_size):
                for s in range(mb * mb_size, (mb + 1) * mb_size):
                    outputs = X_train[s]
                    for layer in self.layers:
                        outputs = layer.step_forward(outputs)

                    error += self.loss_function(Y_train[s], outputs)

                    d_inputs = self.loss_function(Y_train[s], outputs, prime=True)
                    for layer in reversed(self.layers):
                        d_inputs = layer.step_backward(d_inputs, learning_rate)

                if (verbose) and (mb_size != samples):
                    print(f"Epoch {e + 1}/{epochs}: mini batch {mb + 1} done")
            for s in range(samples - (samples % mb_size), samples):
                outputs = X_train[s]
                for layer in self.layers:
                    outputs = layer.step_forward(outputs)

                error += self.loss_function(Y_train[s], outputs)

                d_inputs = self.loss_function(Y_train[s], outputs, prime=True)
                for layer in reversed(self.layers):
                    d_inputs = layer.step_backward(d_inputs, learning_rate)

            if (verbose) and (mb_size != samples):
                print(
                    f"Epoch {e + 1}/{epochs}: mini batch {samples // mb_size + 1} done"
                )

            error /= samples
            if verbose:
                print(f"Epoch {e + 1}/{epochs}: Error = {error}")

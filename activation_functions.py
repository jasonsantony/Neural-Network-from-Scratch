import numpy as np


# Hyperbolic tangent and derivative
def tanh(Z: np.ndarray, prime=False):
    T = np.tanh(Z)
    if not prime:
        return T
    return 1 - T * T


# Sigmoid function and derivative
def sigmoid(Z: np.ndarray, prime=False):
    if not prime:
        return 1.0 / (1.0 + np.exp(-Z))
    sig = 1.0 / (1.0 + np.exp(-Z))
    return sig * (1.0 - sig)


# ReLU function and derivative
def relu(Z: np.ndarray, prime=False):
    if not prime:
        return np.maximum(0, Z)
    return (Z > 0) * 1.0

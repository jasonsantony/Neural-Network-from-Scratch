import numpy as np


# Calculate cost and its derivative with mean squared error
def mse(Y: np.ndarray, Y_hat: np.ndarray, prime=False):
    if not prime:
        return np.mean(np.power(Y - Y_hat, 2))

    return 2.0 * (Y_hat - Y) / Y.size


# Calculate cost and its derivative with cross entropy
def ce(Y: np.ndarray, Y_hat: np.ndarray, prime=False):
    eps = np.finfo(np.float64).eps
    if not prime:
        term_1 = (1.0 - Y) * np.log(1.0 - Y_hat + eps)
        term_2 = Y * np.log(Y_hat + eps)
        return -np.mean(term_1 + term_2)

    return -((Y) / (Y_hat + eps)) + ((1.0 - Y) / (1.0 - Y_hat + eps))

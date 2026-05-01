import numpy as np


def sigmoid(x):
    """Sigmoid activation function.

    Step-by-step:
    1. Accept scalar/array-like input `x` and rely on numpy broadcasting.
    2. Compute exponent `exp(-x)`.
    3. Compute `1 / (1 + exp(-x))` elementwise and return the result.
    """
    return 1 / (1 + np.exp(-x))


def relu(x):
    """ReLU activation function.

    Step-by-step:
    1. Accept scalar/array-like input `x`.
    2. Apply elementwise `max(0, x)` using numpy's vectorized `maximum`.
    3. Return the result (non-negative values with negative inputs clipped to 0).
    """
    return np.maximum(0, x)


ReLU = relu


if __name__ == "__main__":
    arr = np.array([-2, -1, 0, 1, 2])
    print("Input:", arr)
    print("ReLU:", relu(arr))
    print("Sigmoid:", sigmoid(arr))
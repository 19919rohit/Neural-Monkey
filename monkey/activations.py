# monkey/activations.py

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

def relu(x):
    if HAS_NUMPY and isinstance(x, np.ndarray):
        return np.maximum(0, x)
    return max(0, x)

def relu_deriv(x):
    if HAS_NUMPY and isinstance(x, np.ndarray):
        return (x > 0).astype(float)
    return 1.0 if x > 0 else 0.0

def linear(x):
    if HAS_NUMPY and isinstance(x, np.ndarray):
        return x
    return x

def linear_deriv(x):
    if HAS_NUMPY and isinstance(x, np.ndarray):
        return np.ones_like(x)
    return 1.0

def sigmoid(x):
    if HAS_NUMPY and isinstance(x, np.ndarray):
        return 1 / (1 + np.exp(-x))
    return 1 / (1 + np.exp(-x))

def sigmoid_deriv(x):
    s = sigmoid(x)
    if HAS_NUMPY and isinstance(x, np.ndarray):
        return s * (1 - s)
    return s * (1 - s)

def tanh(x):
    if HAS_NUMPY and isinstance(x, np.ndarray):
        return np.tanh(x)
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

def tanh_deriv(x):
    t = tanh(x)
    if HAS_NUMPY and isinstance(x, np.ndarray):
        return 1 - t ** 2
    return 1 - t * t

activation_map = {
    "relu": (relu, relu_deriv),
    "linear": (linear, linear_deriv),
    "sigmoid": (sigmoid, sigmoid_deriv),
    "tanh": (tanh, tanh_deriv)
}
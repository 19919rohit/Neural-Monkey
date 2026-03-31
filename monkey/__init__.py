# monkey/__init__.py

# --- Globals ---
from .globals import useNumpy

# --- Neural Network ---
from .nn import NeuralNet, Dense

# --- Activations ---
from .activations import relu, sigmoid, tanh, linear, activation_map

# --- Attention ---
from .attention import AttentionBlock

# --- Optimizers ---
from .optimizers import SGD, Adam, RMSProp, AdaGrad

# --- Models (Saving/Loading) ---
from .models import save, load

# --- Version ---
__version__ = "1.0.0"

# --- API Exposure ---
__all__ = [
    "useNumpy",
    "NeuralNet",
    "Dense",
    "relu",
    "sigmoid",
    "tanh",
    "linear",
    "activation_map",
    "AttentionBlock",
    "SGD",
    "Adam",
    "RMSProp",
    "AdaGrad", 
    "save",
    "load",
]

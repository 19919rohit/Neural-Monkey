# monkey/models.py
import pickle
import os
from . import nn
from . import globals

# --- NumPy safe flag ---
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
globals.useNumpy = HAS_NUMPY


def save(model, filename):
    """
    Save a NeuralNet model to a .mon file (full model including weights, biases, and activations).
    """
    ext = os.path.splitext(filename)[1].lower()
    if ext != ".mon":
        raise ValueError("Only .mon format is supported")

    data = {
        "input_size": model.input_size,
        "layers": [],
        "optimizer": model.optimizer_name,
        "learning_rate": model.learning_rate,
        "total_parameters": model.total_parameters
    }

    for layer in model.layers:
        layer_data = {
            "input_size": layer.weights.shape[0] if globals.useNumpy else len(layer.weights),
            "output_size": layer.weights.shape[1] if globals.useNumpy else len(layer.biases),
            "activation": None,
            "weights": layer.weights,
            "biases": layer.biases,
            "is_output": layer.is_output
        }

        # Save activation name
        for name, (act, deriv) in nn.activation_map.items():
            try:
                if layer.activation.__code__ == act.__code__:
                    layer_data["activation"] = name
                    break
            except AttributeError:
                continue

        data["layers"].append(layer_data)

    with open(filename, "wb") as f:
        pickle.dump(data, f)

    print(f"Model saved to {filename}")


def load(filename, use_numpy=True):
    """
    Load a NeuralNet model from a .mon file.
    """
    if not filename.endswith(".mon"):
        raise ValueError("Model file must end with .mon extension")

    globals.useNumpy = use_numpy and HAS_NUMPY

    with open(filename, "rb") as f:
        data = pickle.load(f)

    model = nn.NeuralNet(
        input_size=data["input_size"],
        lr=data.get("learning_rate", 0.01),
        optimizer=data.get("optimizer", "sgd")
    )

    for layer_data in data["layers"]:
        act_name = layer_data.get("activation", "linear")
        act, act_deriv = nn.activation_map.get(act_name, (lambda x: x, lambda x: 1))

        dense = nn.Dense(layer_data["input_size"], layer_data["output_size"], act, act_deriv)
        dense.weights = layer_data["weights"]
        dense.biases = layer_data["biases"]
        dense.is_output = layer_data.get("is_output", False)

        model.layers.append(dense)

    # Update total_parameters after loading
    model._update_total_parameters()

    return model
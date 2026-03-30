# monkey/nn.py
import random
from . import globals
from . import optimizers
from .activations import activation_map

# --- NumPy safe import ---
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

globals.useNumpy = HAS_NUMPY


class Dense:
    _counter = 0

    def __init__(self, input_size, output_size, activation, activation_deriv):
        self.layer_id = Dense._counter
        Dense._counter += 1
        self.input_size = input_size
        self.output_size = output_size

        if globals.useNumpy and HAS_NUMPY:
            self.weights = np.random.uniform(-0.5, 0.5, (input_size, output_size))
            self.biases = np.zeros(output_size)
        else:
            self.weights = [[random.uniform(-0.5, 0.5) for _ in range(output_size)] for _ in range(input_size)]
            self.biases = [0] * output_size

        self.activation = activation
        self.activation_deriv = activation_deriv
        self.last_input = None
        self.last_output = None
        self.is_output = False
        self._train_last = None
        self.num_parameters = input_size * output_size + output_size

    def forward(self, x):
        # Ensure x is 2D for batch handling
        if globals.useNumpy and HAS_NUMPY:
            self.last_input = np.array(x, ndmin=2)
            z = self.last_input @ self.weights + self.biases
            self.last_output = self.activation(z)
        else:
            self.last_input = x
            z = [sum(self.last_input[i] * self.weights[i][j] for i in range(len(self.last_input))) + self.biases[j]
                 for j in range(len(self.weights[0]))]
            self.last_output = [self.activation(v) for v in z]
        return self.last_output

    def backward(self, grad_output, lr, optimizer=None):
        # Handle NumPy path
        if globals.useNumpy and HAS_NUMPY:
            grad = np.array(grad_output, ndmin=2) * self.activation_deriv(self.last_output)
            grad_input = grad @ self.weights.T

            # Selective parameter update if _train_last is set
            delta_w = self.last_input.T @ grad
            delta_b = grad.sum(axis=0)

            if self._train_last is not None:
                flat_w = delta_w.flatten()
                flat_b = delta_b.flatten()
                train_count = min(self._train_last, flat_w.size + flat_b.size)
                # Update only the last N parameters
                if train_count > 0:
                    w_count = min(train_count, flat_w.size)
                    b_count = train_count - w_count
                    flat_w[-w_count:] = flat_w[-w_count:]  # keep only last w_count
                    flat_b[-b_count:] = flat_b[-b_count:]  # keep only last b_count
                    delta_w = flat_w.reshape(delta_w.shape)
                    delta_b = flat_b.reshape(delta_b.shape)

            if optimizer:
                self.weights = optimizer.step(f"w_{self.layer_id}", self.weights, delta_w)
                self.biases = optimizer.step(f"b_{self.layer_id}", self.biases, delta_b)
            else:
                self.weights -= lr * delta_w
                self.biases -= lr * delta_b

            return grad_input[0] if grad_input.shape[0] == 1 else grad_input

        # Plain Python lists path
        else:
            grad_input = [0] * len(self.last_input)
            for i in range(len(self.last_input)):
                for j in range(len(self.weights[0])):
                    g = grad_output[j] * self.activation_deriv(self.last_output[j])
                    grad_input[i] += g * self.weights[i][j]

                    if self._train_last is None or (i * len(self.weights[0]) + j) >= (self.input_size * self.output_size - self._train_last):
                        if optimizer:
                            self.weights[i][j] = optimizer.step(f"w_{self.layer_id}_{i}_{j}", self.weights[i][j], g * self.last_input[i])
                            self.biases[j] = optimizer.step(f"b_{self.layer_id}_{j}", self.biases[j], g)
                        else:
                            self.weights[i][j] -= lr * g * self.last_input[i]
                            self.biases[j] -= lr * g
            return grad_input


class NeuralNet:
    def __init__(self, input_size=None, lr=0.01, optimizer="sgd"):
        self.layers = []
        self.input_size = input_size
        self.learning_rate = lr
        self.optimizer_name = optimizer
        self.optimizer = self._init_optimizer(optimizer)
        self.total_parameters = 0

    def _init_optimizer(self, name):
        name = name.lower()
        if name == "adam":
            return optimizers.Adam(lr=self.learning_rate)
        elif name == "rmsprop":
            return optimizers.RMSProp(lr=self.learning_rate)
        elif name == "adagrad":
            return optimizers.AdaGrad(lr=self.learning_rate)
        else:
            return optimizers.SGD(lr=self.learning_rate)

    def add_layer(self, neurons, activation='relu', layer='hidden', input_size=None):
        if not self.layers:
            input_size = input_size or self.input_size
            if input_size is None:
                raise ValueError("Input size must be specified for the first layer")
        else:
            input_size = len(self.layers[-1].biases)

        act, act_deriv = activation_map.get(activation, (lambda x: x, lambda x: 1))
        dense = Dense(input_size, neurons, act, act_deriv)
        dense.is_output = layer.lower() == "output"
        self.layers.append(dense)
        self._update_total_parameters()

    def _update_total_parameters(self):
        self.total_parameters = sum(layer.num_parameters for layer in self.layers)

    def predict(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def train(self, x_train, y_train=None, epochs=1000, shuffle=True, verbose=100, lr=None, next_step=False, optimizer=None):
        learning_rate = lr or self.learning_rate

        if isinstance(optimizer, str):
            self.optimizer = self._init_optimizer(optimizer)
        elif optimizer:
            self.optimizer = optimizer

        for epoch in range(epochs):
            total_loss = 0
            data = x_train
            targets = y_train

            if next_step:
                data = [x_train[i:i+2] for i in range(len(x_train)-2)]
                targets = [x_train[i+2] for i in range(len(x_train)-2)]

            # autoencoder fallback
            if targets is None:
                combined = [(x, x) for x in data]
            else:
                combined = list(zip(data, targets))

            if shuffle:
                random.shuffle(combined)

            for x, y in combined:
                x_list = x if isinstance(x, list) else [x]
                y_list = y if isinstance(y, list) else [y]

                if globals.useNumpy and HAS_NUMPY:
                    x_list = np.array(x_list, ndmin=2)
                    y_list = np.array(y_list, ndmin=2)

                output = self.predict(x_list)

                if globals.useNumpy and HAS_NUMPY:
                    grad = 2 * (output - y_list)
                    total_loss += float(np.sum((output - y_list) ** 2))
                else:
                    grad = [2 * (o - t) for o, t in zip(output, y_list)]
                    total_loss += sum((o - t) ** 2 for o, t in zip(output, y_list))

                for layer in reversed(self.layers):
                    grad = layer.backward(grad, learning_rate, self.optimizer)

            if verbose and epoch % verbose == 0:
                print(f"Epoch {epoch}, Loss: {total_loss:.6f}")
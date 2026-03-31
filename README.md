# Monkey – Simple Neural Networks for Beginners
![Monkey Logo](assets/Neural-Monkey.png)
Monkey is a lightweight Python library for building, training, and experimenting with simple neural networks.
It’s designed for beginners to explore layers, activations, forward/backpropagation, and attention mechanisms.

---

## Features

- Fully connected neural networks (Dense layers)
- Activation functions: ReLU, Sigmoid, Tanh, Linear
- Train networks using gradient descent with configurable optimizers
- Make predictions on new inputs, sequences, or autoencoder-style data
- Lightweight attention block for sequence inputs
- Works with Python lists or NumPy arrays if available
- Beginner-friendly API with minimal setup

---

## Installation

```bash
pip install monkey
```

---

## Quick Start Examples

### 1. Predict the sum of two numbers

```python
from monkey.nn import NeuralNet

x_train = [[2, 8], [9, 3], [7, 4], [1, 1]]
y_train = [[sum(pair)] for pair in x_train]

nn = NeuralNet(input_size=2)
nn.add_layer(neurons=5, activation='relu')
nn.add_layer(neurons=1, activation='relu', layer='output')

nn.train(x_train, y_train, epochs=500, lr=0.1)
print(nn.predict([3,5])[0])
```

### 2. Using Sigmoid activation

```python
nn = NeuralNet(input_size=2)
nn.add_layer(neurons=4, activation='sigmoid')
nn.add_layer(neurons=1, activation='sigmoid', layer='output')

nn.train(x_train, y_train, epochs=1000, lr=0.05)
print(nn.predict([2,2])[0])
```

### 3. Using AttentionBlock for sequences

```python
from monkey.attention import AttentionBlock

seq_input = [[0.8, 0.2, 0.1], [0.5, 0.1, 0.3], [0.2, 0.7, 0.6]]
attn = AttentionBlock(input_size=3, output_size=3)
seq_output = attn.forward(seq_input)
print(seq_output)
```

---

## Model Saving and Loading

```python
from monkey.models import save, load

save(nn, "my_model.mon")
loaded_model = load("my_model.mon", use_numpy=True)
```

---

## Available APIs

**Module: monkey.nn**
- NeuralNet       : Core class for creating and training networks
- Dense           : Individual dense layer

**Module: monkey.activations**
- relu            : ReLU activation
- sigmoid         : Sigmoid activation
- tanh            : Tanh activation
- linear          : Linear activation
- activation_map  : Dictionary of activation functions

**Module: monkey.attention**
- AttentionBlock  : Lightweight attention for sequences

**Module: monkey.optimizers**
- SGD             : Stochastic Gradient Descent
- Adam            : Adam optimizer
- RMSProp         : RMSProp optimizer
- AdaGrad         : AdaGrad optimizer

**Module: monkey.models**
- save            : Save a NeuralNet to a .mon file
- load            : Load a NeuralNet from a .mon file

**Global Flags**
- useNumpy        : Enable or disable NumPy usage (True/False)

---
## Full API Table

# ================= Monkey Library – Full API Reference =================

## Core Classes

| Class / Function           | Description                                                                 |
|----------------------------|-----------------------------------------------------------------------------|
| NeuralNet                  | Create and train fully connected neural networks                           |
| NeuralNet(input_size=None, lr=0.01, optimizer='sgd') | Initialize a network. `input_size` required for first layer. `lr` sets learning rate. `optimizer` can be `'sgd'`, `'adam'`, `'rmsprop'`, or `'adagrad'` |
| NeuralNet.add_layer(neurons, activation='relu', layer='hidden', input_size=None) | Add a layer to the network. `neurons` = number of neurons, `activation` = `'relu'`, `'sigmoid'`, `'tanh'`, or `'linear'`, `layer` = `'hidden'` or `'output'`. `input_size` only for first layer |
| NeuralNet.train(x_train, y_train=None, epochs=1000, shuffle=True, verbose=100, lr=None, next_step=False, optimizer=None) | Train the network. Pass `y_train=None` for autoencoder-style training. `next_step=True` enables sequence-style prediction. `optimizer` can be a string or custom optimizer |
| NeuralNet.predict(x)       | Run forward pass and get predictions for a single input or batch          |
| Dense                      | Fully connected layer (used internally; users interact via NeuralNet)     |

## Attention

| Class / Function           | Description                                                                 |
|----------------------------|-----------------------------------------------------------------------------|
| AttentionBlock(input_size, output_size) | Simple attention mechanism for sequence inputs. Supports lists or NumPy arrays |
| AttentionBlock.forward(X)  | Forward pass to compute attention output for a sequence                     |

## Activations

| Function                   | Description                                                                 |
|----------------------------|-----------------------------------------------------------------------------|
| relu(x)                    | Rectified Linear Unit activation                                            |
| sigmoid(x)                 | Sigmoid activation                                                         |
| tanh(x)                    | Tanh activation                                                            |
| linear(x)                  | Linear activation                                                          |
| activation_map             | Dictionary mapping activation names (`'relu'`, `'sigmoid'`, `'tanh'`, `'linear'`) to functions and derivatives |

## Optimizers

| Class / Function           | Description                                                                 |
|----------------------------|-----------------------------------------------------------------------------|
| SGD(lr=0.01)               | Stochastic Gradient Descent optimizer                                       |
| Adam(lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8) | Adam optimizer with adjustable hyperparameters                   |
| RMSProp(lr=0.001, beta=0.9, eps=1e-8) | RMSProp optimizer with adjustable hyperparameters                 |
| AdaGrad(lr=0.01, eps=1e-8) | AdaGrad optimizer with adjustable hyperparameters                        |

## Saving & Loading Models

| Function                   | Description                                                                 |
|----------------------------|-----------------------------------------------------------------------------|
| save(model, filename)      | Save a NeuralNet model to a `.mon` file (weights, biases, activations)     |
| load(filename, use_numpy=True) | Load a NeuralNet model from a `.mon` file. `use_numpy=False` forces pure Python mode |

## Global Options

| Variable                   | Description                                                                 |
|----------------------------|-----------------------------------------------------------------------------|
| useNumpy                    | Boolean. If True, NumPy is used for computations; fallback to pure Python if False |

## Notes for Users

- Only `.mon` model format is supported.
- Works with Python lists or NumPy arrays seamlessly.  
- `next_step=True` is useful for sequence prediction tasks.  
- Autoencoder-style training happens automatically if `y_train=None`.  
- Recommended to start with small networks and datasets for testing concepts.

---

## Learning Tips

- Start with a single hidden layer and few neurons
- Use small datasets (like sum of two numbers) for testing
- Adjust `learning_rate` and `epochs` to observe convergence
- Experiment with different activation functions
- Try AttentionBlock for sequence-based learning


## License

MIT License

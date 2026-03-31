# Monkey – Official Documentation

Monkey is a lightweight Python library for building, training, and experimenting with simple neural networks.  
It is designed to help beginners understand core machine learning concepts through a minimal and transparent API.

---

# Overview

Monkey provides tools to:

- Build fully connected neural networks
- Train models using backpropagation
- Experiment with activation functions and optimizers
- Work with both NumPy arrays and pure Python lists
- Apply simple attention mechanisms to sequence data
- Save and load trained models

---

# Installation

```bash
pip install monkey
```

---

# Getting Started

## Creating a Neural Network

```python
from monkey import NeuralNet

model = NeuralNet(input_size=2)
model.add_layer(5, activation="relu")
model.add_layer(1, activation="linear", layer="output")
```

---

## Training the Model

```python
x_train = [[2, 8], [9, 3], [7, 4], [1, 1]]
y_train = [[10], [12], [11], [2]]

model.train(x_train, y_train, epochs=500, lr=0.1)
```

---

## Making Predictions

```python
result = model.predict([3, 5])
print(result)
```

---

# Core Concepts

## NeuralNet

The `NeuralNet` class is the main interface for building and training models.

### Parameters

- `input_size` (int): Required for the first layer  
- `lr` (float): Learning rate (default: 0.01)  
- `optimizer` (str): `"sgd"`, `"adam"`, `"rmsprop"`, `"adagrad"`  

---

## Adding Layers

```python
model.add_layer(
    neurons=5,
    activation="relu",
    layer="hidden"
)
```

### Arguments

- `neurons`: Number of neurons  
- `activation`: `"relu"`, `"sigmoid"`, `"tanh"`, `"linear"`  
- `layer`: `"hidden"` or `"output"`  
- `input_size`: Only needed for the first layer  

---

## Training

```python
model.train(
    x_train,
    y_train,
    epochs=1000,
    shuffle=True,
    verbose=100,
    lr=None,
    next_step=False,
    optimizer=None
)
```

### Behavior

- If `y_train` is `None` → autoencoder mode  
- If `next_step=True` → sequence prediction mode  
- Supports both lists and NumPy arrays  

---

## Prediction

```python
model.predict(x)
```

Runs a forward pass through all layers.

---

# Training Modes

## Supervised Learning

```python
model.train(x_train, y_train)
```

Standard input-output training.

---

## Autoencoder Mode

```python
model.train(data)
```

Learns to reconstruct input automatically.

---

## Sequence Prediction

```python
model.train(sequence, next_step=True)
```

Learns to predict the next value in a sequence.

---

# Activation Functions

Available activations:

- `relu`
- `sigmoid`
- `tanh`
- `linear`

Example:

```python
model.add_layer(4, activation="sigmoid")
```

---

# Optimizers

Supported optimizers:

- SGD
- Adam
- RMSProp
- AdaGrad

### Example

```python
model = NeuralNet(2, optimizer="adam")
```

Or during training:

```python
model.train(x, y, optimizer="rmsprop")
```

---

# AttentionBlock

The `AttentionBlock` provides a simple attention mechanism for sequence inputs.

## Example

```python
from monkey import AttentionBlock

attn = AttentionBlock(input_size=3, output_size=3)

sequence = [
    [0.8, 0.2, 0.1],
    [0.5, 0.1, 0.3],
    [0.2, 0.7, 0.6]
]

output = attn.forward(sequence)
```

---

# Model Saving and Loading

## Save Model

```python
from monkey import save

save(model, "model.mon")
```

## Load Model

```python
from monkey import load

model = load("model.mon", use_numpy=True)
```

---

# Global Configuration

## useNumpy

Controls whether NumPy is used for computation.

```python
from monkey import useNumpy

useNumpy = True
```

- If NumPy is not available → automatically falls back to Python lists  
- Setting `useNumpy=False` forces pure Python mode  

---

# Internal Behavior

- Uses forward propagation + backpropagation  
- Mean Squared Error (MSE) loss  
- Batch size = 1 (stochastic training)  
- Supports both scalar and vector inputs  

---

# Limitations

- Only fully connected (Dense) networks are supported  
- No GPU acceleration  
- No convolutional or recurrent layers  
- `.mon` is the only supported model format  

---

# Best Practices

- Start with small networks  
- Use low learning rates for stability  
- Prefer `relu` for hidden layers  
- Use `linear` for regression outputs  
- Use `adam` optimizer for faster convergence  

---

# All APIs (Quick Reference)

NeuralNet(input_size=None, lr=0.01, optimizer="sgd")  
add_layer(neurons, activation, layer, input_size)  
train(x_train, y_train, ...)  
predict(x)  

Dense(input_size, output_size, activation, activation_deriv)  

relu, sigmoid, tanh, linear  
activation_map  

AttentionBlock(input_size, output_size)  
AttentionBlock.forward(X)  

SGD, Adam, RMSProp, AdaGrad  

save(model, filename)  
load(filename, use_numpy=True)  

useNumpy  

---

# Repository

https://github.com/19919rohit/Neural-Monkey

---

# License

MIT License

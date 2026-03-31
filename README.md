# Monkey – Simple Neural Networks for Beginners
![Neural-Monkey logo](assets/Neural-Monkey.png) 
Monkey is a lightweight Python library for building, training, and experimenting with simple neural networks.  
It is designed for beginners who want to understand how neural networks work internally without heavy dependencies.

---

## Features

- Fully connected neural networks (Dense layers)  
- Activation functions: ReLU, Sigmoid, Tanh, Linear  
- Train networks using gradient descent with multiple optimizers  
- Supports SGD, Adam, RMSProp, and AdaGrad  
- Works with Python lists or NumPy arrays  
- Autoencoder-style training (no labels required)  
- Sequence prediction using next-step training  
- Lightweight AttentionBlock for sequence inputs  
- Save and load models using `.mon` format  
- Minimal and beginner-friendly API  

---

## Installation

```bash
pip install monkey
```

---

## Quick Start

### Predict the sum of two numbers

```python
from monkey import NeuralNet

x_train = [[2, 8], [9, 3], [7, 4], [1, 1]]
y_train = [[sum(pair)] for pair in x_train]

nn = NeuralNet(input_size=2)
nn.add_layer(neurons=5, activation='relu')
nn.add_layer(neurons=1, activation='relu', layer='output')

nn.train(x_train, y_train, epochs=500, lr=0.1)

print(nn.predict([3, 5])[0])
```

---

### Using different optimizer

```python
from monkey import NeuralNet

x_train = [[2, 8], [9, 3], [7, 4], [1, 1]]
y_train = [[sum(pair)] for pair in x_train]

nn = NeuralNet(input_size=2, optimizer="adam")

nn.add_layer(4, activation="relu")
nn.add_layer(1, activation="linear", layer="output")

nn.train(x_train, y_train, epochs=500)

print(nn.predict([3, 5])[0])
```

---

### Autoencoder (no labels)

```python
from monkey import NeuralNet

data = [[0], [1], [2], [3], [4], [5]]

nn = NeuralNet(input_size=1)
nn.add_layer(3, activation="relu")
nn.add_layer(1, activation="linear", layer="output")

nn.train(data, epochs=200)

print(nn.predict([2]))
```

---

### Sequence prediction (next-step learning)

```python
from monkey import NeuralNet

sequence = [1, 2, 3, 4, 5, 6]

nn = NeuralNet(input_size=1)
nn.add_layer(5, activation="relu")
nn.add_layer(1, activation="linear", layer="output")

nn.train(sequence, epochs=300, next_step=True)

print(nn.predict([6]))
```

---

### AttentionBlock example

```python
from monkey import AttentionBlock

seq_input = [
    [0.8, 0.2, 0.1],
    [0.5, 0.1, 0.3],
    [0.2, 0.7, 0.6]
]

attn = AttentionBlock(input_size=3, output_size=3)
output = attn.forward(seq_input)

print(output)
```

---

## Model Saving and Loading

```python
from monkey import save, load

save(nn, "model.mon")

loaded = load("model.mon", use_numpy=True)

print(loaded.predict([3, 5]))
```

---

## Available API (Public)

### Core
- NeuralNet → Create and train networks  
- Dense → Internal fully connected layer  

### Activations
- relu  
- sigmoid  
- tanh  
- linear  
- activation_map  

### Attention
- AttentionBlock  

### Optimizers
- SGD  
- Adam  
- RMSProp  
- AdaGrad  

### Models
- save  
- load  

### Global
- useNumpy → Toggle NumPy usage (True / False)

---

## Notes

- Only `.mon` model format is supported  
- Works with both Python lists and NumPy arrays  
- If NumPy is unavailable, pure Python mode is used  
- `next_step=True` enables sequence learning  
- If `y_train=None`, autoencoder training is used automatically  

---

## Learning Tips

- Start with small datasets  
- Use fewer neurons to understand behavior  
- Try different activations to observe changes  
- Experiment with optimizers  
- Use AttentionBlock for sequence understanding  

---

## License

MIT License

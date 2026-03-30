from monkey.nn import NeuralNet

# Training data: simple sums
x_train = [[0, 0], [0, 1], [1, 0], [1, 1], [2, 3], [3, 4], [5, 2], [6, 1]]
y_train = [[sum(pair)] for pair in x_train]

# Create the neural network
nn = NeuralNet(input_size=2)

# Add one hidden layer with 5 neurons, ReLU activation
nn.add_layer(neurons=5, activation='relu')

# Output layer with 1 neuron (the sum), ReLU works fine here
nn.add_layer(neurons=1, activation='relu', layer='output')

# Train the network
nn.train(x_train, y_train, epochs=1000, lr=0.01)

# Test predictions
test_cases = [[1, 2], [3, 5], [4, 4], [6, 7]]
for case in test_cases:
    print(f"Prediction for {case}: {nn.predict(case)[0]}")
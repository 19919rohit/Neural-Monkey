"""
Monkey Library - XOR Test Script with Colors
Author: NEUNIX STUDIOS
Purpose: Demonstrates a neural network learning the XOR pattern with colored output.
This version adds detailed comments explaining each step.
"""
import sys, os

# Add parent directory to sys.path so we can import the moneky library
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from monkey.nn import NeuralNet
from monkey.activations import activation_map

# -----------------------------
# ANSI Color Codes
# -----------------------------
# Used to make the console output visually appealing
CYAN = '\033[96m'    # Input vectors
GREEN = '\033[92m'   # Correct predictions
YELLOW = '\033[93m'  # Section headers
RED = '\033[91m'     # Errors or wrong predictions
RESET = '\033[0m'    # Reset to default color

# -----------------------------
# 1. XOR Dataset
# -----------------------------
# Training data for XOR
# Inputs are 2-bit binary numbers
# Output is the XOR of the input bits
x_train = [
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
]
y_train = [
    [0],
    [1],
    [1],
    [0]
]

# -----------------------------
# 2. Create Neural Network
# -----------------------------
# Initialize neural network with 2 inputs
nn = NeuralNet(input_size=2, lr=0.1)

# Add hidden layer
# Essential for solving XOR since XOR is not linearly separable
nn.add_layer(neurons=4, activation='tanh')  # Hidden layer with 4 neurons

# Add output layer
# Single neuron with sigmoid activation to predict value between 0 and 1
nn.add_layer(neurons=1, activation='sigmoid', layer='output')

# -----------------------------
# 3. Predictions Before Training
# -----------------------------
print(f"{YELLOW}-- Predictions Before Training --{RESET}")
for x, y in zip(x_train, y_train):
    pred = nn.predict(x)[0]  # Network prediction
    print(f"{CYAN}{x}{RESET} -> predicted: {GREEN}{pred:.2f}{RESET}, actual: {CYAN}{y[0]}{RESET}")

# -----------------------------
# 4. Train Neural Network
# -----------------------------
print(f"\n{YELLOW}-- Training Neural Network --{RESET}")
# Train the network for 5000 epochs, printing loss every 500 epochs
nn.train(x_train, y_train, epochs=5000, verbose=500)

# -----------------------------
# 5. Predictions After Training
# -----------------------------
print(f"\n{YELLOW}-- Predictions After Training --{RESET}")
for x, y in zip(x_train, y_train):
    pred = nn.predict(x)[0]        # Predicted value
    rounded = round(pred)           # Rounded value to 0 or 1
    # Color prediction green if correct, red if wrong
    color_pred = GREEN if rounded == y[0] else RED
    print(f"{CYAN}{x}{RESET} -> predicted: {color_pred}{pred:.2f}{RESET} actual: {CYAN}{y[0]}{RESET}")

# -----------------------------
# 6. Interactive XOR Mode
# -----------------------------
print(f"\n{YELLOW}-- Interactive XOR Mode --{RESET}")
print(f"Enter two binary numbers (0 or 1) separated by space, or 'exit' to quit.")

while True:
    user_input = input(">> ").strip()
    if user_input.lower() == 'exit':
        break
    try:
        # Convert input into list of integers
        nums = list(map(int, user_input.split()))
        # Validate input
        if len(nums) != 2 or any(n not in [0, 1] for n in nums):
            print(f"{RED}Please enter exactly two binary numbers (0 or 1).{RESET}")
            continue
        # Predict XOR
        prediction = nn.predict(nums)[0]
        rounded = round(prediction)
        # Display prediction and rounded output with colors
        print(f"{CYAN}{nums}{RESET} -> Predicted XOR: {GREEN}{prediction:.2f}{RESET} -> Rounded: {GREEN}{rounded}{RESET}")
    except Exception as e:
        print(f"{RED}Error: {e}{RESET}")
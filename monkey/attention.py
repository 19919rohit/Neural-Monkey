# monkey/attention.py
import random

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

class AttentionBlock:
    """
    Simplified attention block for sequence inputs.
    Supports both NumPy arrays and plain Python lists.
    """

    class Layer:
        def __init__(self, input_size, output_size):
            if HAS_NUMPY:
                self.weights = np.random.uniform(-0.5, 0.5, (input_size, output_size))
                self.biases = np.zeros(output_size)
            else:
                self.weights = [[random.uniform(-0.5, 0.5) for _ in range(output_size)]
                                for _ in range(input_size)]
                self.biases = [0] * output_size

        def forward(self, x):
            if HAS_NUMPY:
                x_arr = np.array(x)
                return x_arr @ self.weights + self.biases
            else:
                return [sum(x[i] * self.weights[i][j] for i in range(len(x))) + self.biases[j]
                        for j in range(len(self.weights[0]))]

    def __init__(self, input_size, output_size):
        self.Wq = self.Layer(input_size, output_size)
        self.Wk = self.Layer(input_size, output_size)
        self.Wv = self.Layer(input_size, output_size)

    def forward(self, X):
        if HAS_NUMPY:
            X_arr = np.array(X)
            Q = X_arr @ self.Wq.weights + self.Wq.biases
            K = X_arr @ self.Wk.weights + self.Wk.biases
            V = X_arr @ self.Wv.weights + self.Wv.biases

            # Attention scores: Q @ K.T
            scores = Q @ K.T
            # Normalize rows
            row_sums = scores.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1  # avoid division by zero
            attn_weights = scores / row_sums
            # Weighted sum of V
            output = attn_weights @ V
            return output.tolist()
        else:
            # Python lists fallback
            Q = [self.Wq.forward(x) for x in X]
            K = [self.Wk.forward(x) for x in X]
            V = [self.Wv.forward(x) for x in X]

            # Compute attention scores
            attn_scores = [[sum(qi * kj for qi, kj in zip(q, k)) for k in K] for q in Q]

            # Normalize scores
            attn_weights = [[s / sum(row) if sum(row) != 0 else 1 / len(row) for s in row]
                            for row in attn_scores]

            # Compute weighted output
            output = []
            for w_row in attn_weights:
                out_vec = [0] * len(V[0])
                for i, w in enumerate(w_row):
                    for j, v_val in enumerate(V[i]):
                        out_vec[j] += w * v_val
                output.append(out_vec)
            return output
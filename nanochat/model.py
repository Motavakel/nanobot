import random

class NanoChatModel:
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        self.weights = [[random.uniform(-1, 1) for _ in range(input_size)] for _ in range(output_size)]
        self.biases = [random.uniform(-1, 1) for _ in range(output_size)]

    def forward(self, x):
        result = []
        for i in range(self.output_size):
            total = sum(w * xi for w, xi in zip(self.weights[i], x))
            total += self.biases[i]
            result.append(total)
        return result

    def train(self, X, y, lr=0.1, epochs=1000):
        for _ in range(epochs):
            for xi, target_idx in zip(X, y):
                out = self.forward(xi)
                pred_idx = out.index(max(out))
                if pred_idx != target_idx:
                    for j in range(self.input_size):
                        self.weights[target_idx][j] += lr * xi[j]
                        self.weights[pred_idx][j] -= lr * xi[j]

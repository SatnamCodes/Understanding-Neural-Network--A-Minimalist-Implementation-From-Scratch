import numpy as np
import pandas as pd
from preprocess import load_preprocess_data

np.random.seed(42)
class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01, epochs=1000):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.epochs = epochs

        self.W1 = np.random.rand(self.input_size, self.hidden_size)
        self.B1 = np.zeros((1, self.hidden_size))
        self.W2 = np.random.randn(self.hidden_size, self.output_size)
        self.B2 = np.zeros((1, self.output_size))

        self.loss = []

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def sigmoid_derivate(self, a):
        return a * (1 - a)

    def relu(self, z):
        return np.maximum(0, z)

    def relu_derivative(self, z):
        return (z > 0).astype(float)

    def forward_prop(self, x):
        self.z1 = np.dot(x, self.W1) + self.B1
        self.a1 = self.sigmoid(self.z1)

        self.z2 = np.dot(self.a1, self.W2) + self.B2
        self.a2 = self.sigmoid(self.z2)

        return self.a2

    def compute_loss(self, y_pred, y):
        eps = 1e-8  # avoid log(0)
        return -np.mean(y * np.log(y_pred + eps) + (1 - y) * np.log(1 - y_pred + eps))


    def backward_prop(self, x, y):
        m = y.shape[0]

        # Output layer
        dZ2 = self.a2 - y 
        dW2 = np.dot(self.a1.T, dZ2) / m
        dB2 = np.sum(dZ2, axis=0, keepdims=True) / m


        # Hidden layer
        dA1 = np.dot(dZ2, self.W2.T)
        dZ1 = dA1 * self.sigmoid_derivate(self.a1)
        dW1 = np.dot(x.T, dZ1) / m
        dB1 = np.sum(dZ1, axis=0, keepdims=True) / m

        self.W2 -= self.learning_rate * dW2
        self.B2 -= self.learning_rate * dB2
        self.W1 -= self.learning_rate * dW1
        self.B1 -= self.learning_rate * dB1

    def train(self, x, y):
        for epoch in range(self.epochs):
            y_pred = self.forward_prop(x)
            loss = self.compute_loss(y_pred, y)
            self.loss.append(loss)

            self.backward_prop(x, y)

            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss:.4f}")
                
    
        import matplotlib.pyplot as plt
        plt.plot(self.loss)
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title("Loss Curve")
        plt.show()

    def predict(self, x):
        probs = self.forward_prop(x)
        return (probs >0.5).astype(int)
    
    def accuracy(self, y_pred, y_true):
        return np.mean(y_pred == y_true)

    

# Main Execution
x_train, x_test, y_train, y_test = load_preprocess_data("data/data.csv")
# Initializing the Neural Network
nn = NeuralNetwork(input_size=x_train.shape[1], hidden_size=10, output_size=1, learning_rate=0.01, epochs=1000)
# Training the Neural Network
nn.train(x_train, y_train)
# Making Predictions
pred = nn.predict(x_test)
# Evaluating the Model
accuracy = nn.accuracy(pred, y_test)
print(f"Test Accuracy: {accuracy * 100:.2f}%")
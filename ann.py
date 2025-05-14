import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder

#loading and preprocessing
iris = datasets.load_iris()
X = iris.data
y = iris.target

scaler = StandardScaler()
X = scaler.fit_transform(X)

encoder = OneHotEncoder(sparse_output=False)
y = y.reshape(-1,1)
y = encoder.fit_transform(y)

X_train, X_test ,y_train , y_test = train_test_split(X,y,test_size = 0.2)

#activation functions
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(a):
    return a * (1 - a)

def relu(z):
    return np.maximum(0, z)

def relu_derivative(z):
    return (z > 0).astype(int)

def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

np.random.seed(42)

input_size = X_train.shape[1]
hidden_size = 5
output_size = y_train.shape[1]

W1 = np.random.randn(input_size, hidden_size)
b1 = np.zeros((1, hidden_size))

W2 = np.random.randn(hidden_size, output_size)
b2 = np.zeros((1, output_size))

#training

lr = 0.1
epochs = 10000

for epoch in range(epochs):
    Z1 = np.dot(X_train , W1) + b1
    A1 = relu(Z1)

    Z2 = np.dot(A1 , W2) + b2
    A2 = relu(Z2)

    # Loss calculation (cross-entropy)
    eps = 1e-8  # for numerical stability
    loss = -np.mean(np.sum(y_train * np.log(A2 + eps), axis=1))

    # Backpropagation
    dA2 = A2 - y_train
    dZ2 = dA2
    dW2 = A1.T @ dZ2
    db2 = np.sum(dZ2, axis=0, keepdims=True)

    dA1 = dZ2 @ W2.T
    dZ1 = dA1 * relu_derivative(Z1)
    dW1 = X_train.T @ dZ1
    db1 = np.sum(dZ1, axis=0, keepdims=True)

    # Update weights and biases
    W2 -= lr * dW2
    b2 -= lr * db2
    W1 -= lr * dW1
    b1 -= lr * db1

    if epoch % 1000 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.4f}")

# Testing the model
Z1_test = X_test @ W1 + b1
A1_test = relu(Z1_test)

Z2_test = A1_test @ W2 + b2
A2_test = softmax(Z2_test)

# Convert probabilities to class predictions
predictions = np.argmax(A2_test, axis=1)

# Accuracy
accuracy = np.mean(predictions == np.argmax(y_test, axis=1))
print(f"Test Accuracy: {accuracy * 100:.2f}%")
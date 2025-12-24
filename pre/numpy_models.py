import numpy as np

def relu(A : np.ndarray) -> np.ndarray :
    return np.maximum(0, A)

def sigmoid(A : np.ndarray) -> np.ndarray :
    return 1 / (1 + np.exp(-A))

class TwoLayerMLP:
    def __init__(self, input_dim, hidden_dim, output_dim=1):
        # 随机初始化权重
        self.W1 = np.random.randn(input_dim, hidden_dim) * 0.01
        self.b1 = np.zeros(hidden_dim)
        self.W2 = np.random.randn(hidden_dim, output_dim) * 0.01
        self.b2 = np.zeros(output_dim)
        
        self.cache = {}
        self.grads = {}
    
    def forward(self, X):
        Z1 = np.dot(X, self.W1) + self.b1
        A1 = relu(Z1)
        Z2 = np.dot(A1, self.W2) + self.b2
        A2 = sigmoid(Z2)
        self.cache = {
            "Z1" : Z1,
            "A1" : A1,
            "Z2" : Z2,
            "A2" : A2,
        }
        return A2
    
    def loss(self, Y_true, Y_pred) -> float :
        eps = 1e-7
        Y_pred = np.clip(Y_pred, eps, 1 - eps)
        return -np.mean(Y_true * np.log(Y_pred) + (1 - Y_true) * np.log(1 - Y_pred))
    
    def backward(self, X, Y_true):
        dZ2 = (self.cache["A2"] - Y_true) / X.shape[0]
        dW2 = np.dot(self.cache["A1"].T, dZ2)
        db2 = np.sum(dZ2, axis = 0)
        dA1 = np.dot(dZ2, self.W2.T)
        dZ1 = dA1 * (self.cache["Z1"] > 0)
        dW1 = np.dot(X.T, dZ1)
        db1 = np.sum(dZ1, axis = 0)
        self.grads = {
            "W1" : dW1,
            "b1" : db1,
            "W2" : dW2,
            "b2" : db2
        }
    
    def update_params(self, lr):
        self.W1 -= lr * self.grads['W1']
        self.b1 -= lr * self.grads['b1']
        self.W2 -= lr * self.grads['W2']
        self.b2 -= lr * self.grads['b2']
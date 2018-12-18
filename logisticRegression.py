import numpy as np
import pandas as pd
from Data_Process import get_binaryData
from costFunction import crossEntropy

def sigmoid(z):
    return 1 / ( 1 + np.exp(-z))

def forward(X, W, b):
    return sigmoid(X.dot(W) + b)

def classification_rate(Y, Y_hat):
    return np.mean(Y == Y_hat)

X, Y, X_test, Y_test = get_binaryData()
X = np.concatenate((X, X_test))
Y = np.concatenate((Y, Y_test))
D = X.shape[1]
W = np.random.randn(D)
b = 0

y_hat = forward(X, W, b)
predictions = np.round(y_hat)
print("predications are ", classification_rate(Y, predictions))
print("entropy cost is ", crossEntropy(Y, y_hat))
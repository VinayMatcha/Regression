import numpy as np
import matplotlib.pyplot as plt
from costFunction import crossEntropyVectorized
from Data_Process import get_binaryData, get_data
from logisticRegression import classification_rate, sigmoid, forward


N = 50
D = 50

X = (np.random.random((N , D)) - 0.5 ) * 10
W = np.array([1, 0.5, -0.5] + [0]*(D-3))
Y = np.round(sigmoid(X.dot(W) + np.random.randn(N)*0.5))

costs = []
W_hat = np.random.randn(D)
b = 0
lr = 0.001
lamba = 5
for i in range(1000):
    Y_hat = forward(X, W_hat, b)
    delta = Y_hat - Y
    W_hat -= lr*(X.T.dot(delta) + lamba*np.sign(W_hat))
    cost = crossEntropyVectorized(Y, Y_hat) + lamba * np.mean(np.abs(W_hat))
    costs.append(cost)
plt.plot(costs)
plt.show()

plt.plot(W, label="Original W")
plt.plot(W_hat, label="L1 W")
plt.legend()
plt.savefig("images/L1 regularization logistic")
plt.show()
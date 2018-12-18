import numpy as np
import matplotlib.pyplot as plt
from costFunction import crossEntropyVectorized
from Data_Process import get_binaryData, get_data
from logisticRegression import classification_rate, sigmoid, forward

N = 4
D = 2

X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1],
])
T = np.array([0, 1, 1, 0])
plt.scatter(X[:,0], X[:,1], c = T)
plt.show()
plt.savefig("images/xor")
bias = np.ones((N, 1))
r = (X[:, 0] * X[:, 1]).reshape(N, 1)
X = np.concatenate((bias, r, X), axis=1)


W = np.random.randn(D+2)
Y = sigmoid(X.dot(W))
lr = 0.001
costs = []
for i in range(9000):
    cost = crossEntropyVectorized(T, Y)
    costs.append(cost)
    if i % 500 == 0:
        print(cost)
    W += lr * (X.T.dot(T-Y) - 0.1 * W)
    Y = sigmoid(X.dot(W))

plt.plot(costs)
plt.title("Xor logistic")
plt.savefig("images/Xor logistic")
plt.show()


print("Final w:", W)
print("Final classification rate:", classification_rate(T, np.round(Y)))
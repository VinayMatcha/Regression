import numpy as np
import matplotlib.pyplot as plt
from costFunction import crossEntropyVectorized
from Data_Process import get_binaryData, get_data
from logisticRegression import classification_rate, sigmoid, forward

N = 1000
D = 2
innerRadius = 5
outerRadius = 10

X1 = np.random.randn(N//2) + innerRadius
X2 = np.random.randn(N//2) + outerRadius
theta = 2*np.pi*np.random.random(N//2)
X1 = np.concatenate([[X1 * np.cos(theta)], [X1 * np.sin(theta)]]).T
theta = 2*np.pi*np.random.random(N//2)
X2 = np.concatenate([[X2 * np.cos(theta)], [X2 * np.sin(theta)]]).T
X = np.concatenate([X1, X2])
T = np.array([0]*(N//2) + [1]*(N//2))
plt.scatter(X[:,0], X[:,1], c = T)
plt.show()
plt.savefig("images/donus")

bias = np.ones((N, 1))
r = np.sqrt((X*X).sum(axis=1)).reshape(-1,1)
X = np.concatenate((bias, r, X), axis=1)
# print(X.shape)
W = np.random.randn(D+2)
Y = sigmoid(X.dot(W))
lr = 0.001
costs = []
for i in range(3000):
    cost = crossEntropyVectorized(T, Y)
    costs.append(cost)
    if i % 500 == 0:
        print(cost)
    W += lr * (X.T.dot(T-Y) - 0.1 * W)
    Y = sigmoid(X.dot(W))

plt.plot(costs)
plt.title("DoughNut logistic")
plt.savefig("images/DoughNut logistic")
plt.show()
print(classification_rate(T, np.round(Y)))
import numpy as np
import matplotlib.pyplot as plt

N = 10
D = 3
X = np.zeros((N, D))
X[:, 0] = 1
X[:5, 1] = 1
X[5:, 2] = 1
Y = np.array([0]*5 + [1]*5)
#storing costs to plot in the end
costs = []

W = np.random.randn(D) / np.sqrt(D)
lr = 0.001
for t in range(1000):
    Yhat = X.dot(W)
    diff = Yhat - Y
    W -= lr * X.T.dot(diff)
    mse = diff.dot(diff)/N
    costs.append(mse)

plt.plot(costs)
plt.show()
plt.plot(Yhat, label = "prediction")
plt.plot(Y, label = "original")
plt.legend()
plt.title("Gradient descent plotting")
plt.show()
plt.savefig("Gradient descent plotting")
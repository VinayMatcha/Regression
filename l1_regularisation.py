import numpy as np
import matplotlib.pyplot as plt

N = 50
D = 50

X = (np.random.random((N, D)) - 0.5)*10
W = np.array([1, 0.5, -0.5] + [0]*(D - 3))
Y = X.dot(W) + np.random.randn(N) * 0.5
costs = []
W_l1 = np.random.randn(D) / np.sqrt(D)
lr = 0.001
l1 = 5
for t in range(500):
    Yhat = X.dot(W_l1)
    diff = Yhat - Y
    W_l1 -= lr*(X.T.dot(diff) + l1*np.sign(W_l1))
    mse = diff.dot(diff)/N
    costs.append(mse)

plt.plot(costs)
plt.show()

plt.plot(W, label="True Values")
plt.plot(W_l1, label = "L1 Gardient Descent values")
plt.title("L1 Gardient Descent")
plt.legend()
plt.show()
plt.savefig("L1 Gardient Descent")



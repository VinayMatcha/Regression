import numpy as np
import matplotlib.pyplot as plt

N = 50
X = np.linspace(0, 10, N)
Y = 0.5 * X + np.random.randn(N)
Y[-1] += 30
Y[-2] += 30
plt.scatter(X, Y)
plt.show()

X = np.vstack((np.ones(N), X)).T

w_linear = np.linalg.solve(X.T.dot(X), X.T.dot(Y))
Y_hat = X.dot(w_linear)
plt.scatter(X[:, 1], Y)
plt.plot(X[:,1], Y_hat)
plt.show()

l2 = 1000
w_l2 =  np.linalg.solve(l2*np.eye(2) + X.T.dot(X), X.T.dot(Y))
Y_l2hat = X.dot(w_l2)
plt.scatter(X[:, 1], Y)
plt.plot(X[:,1], Y_hat, label = "maximum likelihood")
plt.plot(X[:,1], Y_l2hat, label = "l2 regulariztion")
plt.legend()
plt.show()
plt.savefig("linear one vs L2 regulariztion")
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

X = []
Y = []
for line in open("data_poly.csv"):
    x, y = line.split(",")
    x = float(x)
    X.append([1, x, x**2])
    Y.append(float(y))

X = np.array(X)
Y = np.array(Y)

plt.scatter(X[:,1], Y)
plt.show()

#calculating weights. It is just a multiple regression

a = np.dot(X.T, X)
b = np.dot(X.T, Y)
w = np.linalg.solve(a, b)
Yhat = np.dot(X, w)

plt.scatter(X[:,1], Y)
plt.plot(sorted(X[:,1]), sorted(Yhat))
plt.title("Polynomial regression")
plt.show()
plt.savefig("PolyLinearRegression")

dif1 = Y - Yhat
dif2 = Y - Y.mean()
rsqr = 1 - dif1.dot(dif1) / dif2.dot(dif2)
print("rsqr error is {}", rsqr)
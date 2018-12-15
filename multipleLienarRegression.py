import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

X = []
Y = []
for line in open("data_2d.csv"):
    x1, x2, y = line.split(",")
    X.append([1, float(x1), float(x2)])
    Y.append(float(y))

X = np.array(X)
Y = np.array(Y)



fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[:,1], X[:,2], Y)
plt.show()

a = np.dot(X.T, X)
b = np.dot(X.T, Y)
w = np.linalg.solve(a, b)
Yhat = np.dot(X, w)


dif1 = Y - Yhat
dif2 = Y - Y.mean()
rsqr = 1 - dif1.dot(dif1)/dif2.dot(dif2)
print("rsqr error is {}", rsqr)
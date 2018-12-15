import numpy as np
import matplotlib.pyplot as plt

X = []
Y = []

for line in open("data_1d.csv"):
    x, y = line.split(",")
    X.append(float(x))
    Y.append(float(y))

X = np.array(X)
Y = np.array(Y)
plt.scatter(X, Y)

#now we will calcullate the equations
denom = X.dot(X) - X.mean()*X.sum()
a = (X.dot(Y) - Y.mean()*X.sum())/denom
b = (Y.mean() * X.dot(X) - X.mean()*X.dot(Y))/denom

Yhat = a*(X) + b
plt.plot(X, Yhat)
plt.xlabel("X values")
plt.ylabel("Values of Y ")
plt.title("Simple Linear regression")
plt.show()
plt.savefig("Simple_Linear_Regression")

#calcualting the rsquare error
dif1 = Y - Yhat
dif2 = Y - Y.mean()
sse = dif1.dot(dif1)
sst = dif2.dot(dif2)
rsqr = 1 - sse/sst
print("rsqr error is {}", rsqr)
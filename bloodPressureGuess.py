import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def regression(X, Y):
    a = np.dot(X.T, X)
    b = np.dot(X.T, Y)
    w = np.linalg.solve(a, b)
    Yhat = np.dot(X, w)
    dif1 = Y - Yhat
    dif2 = Y - Y.mean()
    rsqr = 1 - dif1.dot(dif1) / dif2.dot(dif2)
    return rsqr

# The data (X1, X2, X3) are for each patient.
# X1 = systolic blood pressure
# X2 = age in years
# X3 = weight in pounds
df = pd.read_excel("mlr02.xls")
X = df.values
plt.scatter(X[:,1], X[:,0])
plt.show()
plt.scatter(X[:,2], X[:,0])
plt.show()

df['ones'] = 1
Y = df['X1']
X = df[['X2', 'X3', 'ones']]
X2 = df[['X2', 'ones']]
X3 = df[[ 'X3', 'ones']]

print("rsqr error for X is{} ", regression(X, Y))
print("rsqr error for X2 only is{} ", regression(X2, Y))
print("rsqr error for X3 only is{} ", regression(X3, Y))
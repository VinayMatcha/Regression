import numpy as np
import re
import matplotlib.pyplot as plt


X = []
Y = []
non_decimal = re.compile(r'[^\d]+')

for line in open("moore.csv"):
    r = line.split('\t')
    y = int(non_decimal.sub('', r[1].split('[')[0]))
    x = int(non_decimal.sub('', r[2].split('[')[0]))
    X.append(x)
    Y.append(y)

X = np.array(X)
Y = np.array(Y)
Y = np.log(Y)
plt.scatter(X, Y)
plt.xlabel("Year")
plt.ylabel("Log of number of transistors")
plt.title("Moore's law")
plt.savefig("moores")



#now we will calcullate the equations
denom = X.dot(X) - X.mean()*X.sum()
a = (X.dot(Y) - Y.mean()*X.sum())/denom
b = (Y.mean() * X.dot(X) - X.mean()*X.dot(Y))/denom

Yhat = a*(X) + b
plt.plot(X, Yhat)
plt.ylabel("Log of number of transistors")
plt.title("Moore's law")
plt.show()
plt.savefig("Moore's law Regression")

#calcualting the rsquare error
dif1 = Y - Yhat
dif2 = Y - Y.mean()
sse = dif1.dot(dif1)
sst = dif2.dot(dif2)
rsqr = 1 - sse/sst
print("the r squared error is {}", rsqr)


#chekcing for validaity
print("TIme to doubkle is ", np.log(2)/a, " years ")
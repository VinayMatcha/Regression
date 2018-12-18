import numpy as np
import matplotlib.pyplot as plt
from costFunction import crossEntropyVectorized
from Data_Process import get_binaryData, get_data
from logisticRegression import classification_rate, sigmoid, forward

Xtrain, Ytrain, Xtest, Ytest = get_binaryData()
N, D = Xtrain.shape
W = np.random.randn(D)
b = 0

trainingCosts = []
testingCosts = []
lr = 0.001
lamba = 0.1
for t in range(1000):
    Y_hat = forward(Xtrain, W, b)
    Ytest_hat = forward(Xtest, W, b)
    trainingCosts.append(crossEntropyVectorized(Ytrain, Y_hat))
    testingCosts.append(crossEntropyVectorized(Ytest, Ytest_hat))
    if t%200 == 0:
        print(trainingCosts[-1], testingCosts[-1], t)
    W -= lr * ( Xtrain.T.dot(Y_hat - Ytrain) + lamba * np.sign(W))
    b -= lr * (Y_hat-Ytrain).sum()

print("Final Traing classification rate ", classification_rate(Ytrain, np.round(Y_hat)))
print("Final Testing classification rate ", classification_rate(Ytest, np.round(Ytest_hat)))

legend1 = plt.plot(trainingCosts)
legend2 = plt.plot(testingCosts)
plt.legend(legend1, legend2)
plt.show()
plt.savefig("images/Ecommerece Logistic L1 reg")

import numpy as np
import pandas as pd

def get_data():
    dataFrame = pd.read_csv("ecommerce_data.csv")
    data = dataFrame.values
    np.random.shuffle(data)
    X = data[:, :-1]
    Y = data[:, -1].astype(np.int32)
    #Normalizing the X data
    X[:, 1] = (X[:, 1] - X[:, 1].mean()) / X[:, 1].std()
    X[:, 2] = (X[:, 2] - X[:, 2].mean()) / X[:, 2].std()

    N, D = X.shape
    X2 = np.zeros((N, D+3))
    X2[:, :D-1] = X[:, :D-1]
    for n in range(N):
        t = int(X[n, D-1])
        X2[n, t+D-1] = 1

    # Z = np.zeros((N, 4))
    # Z[np.arange(N), X[:,D-1].astype(np.int32)] = 1
    # assign: X2[:,-4:] = Z

    Xtrain = X2[:-100]
    Ytrain = Y[:-100]
    Xtest = X2[-100:]
    Ytest = Y[-100:]
    for i in (1, 2):
        m = Xtrain[:, i].mean()
        s = Xtrain[:, i].std()
        Xtrain[:, i] = (Xtrain[:, i] - m) / s
        Xtest[:, i] = (Xtest[:, i] - m) / s

    return Xtrain, Ytrain, Xtest, Ytest

def get_binaryData():
    Xtrain, Ytrain, Xtest, Ytest = get_data()
    X2train = Xtrain[Ytrain <=1]
    Y2train = Ytrain[Ytrain <= 1]
    X2test = Xtest[Ytest <= 1]
    Y2test = Ytest[Ytest <= 1]
    return X2train, Y2train, X2test, Y2test
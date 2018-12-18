import numpy as np

def crossEntropy(t, Y):
    cost = 0
    for n in range(Y.shape[0]):
        if t[n] == 1:
            cost -= np.log(Y[n])
        else:
            cost -= np.log(1 - Y[n])
    return cost

def crossEntropyVectorized(t, Y):
    return -np.mean(t*np.log(Y) + (1-t)*np.log(1-Y))
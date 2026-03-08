import numpy as np
import matplotlib.pyplot as plt
import math

alpha = 0.1
beta = 20
epsilon = 0.1

def phi(x, y):
    return math.exp(-beta*(abs(x - y)-epsilon))/(1+math.exp(-beta*(abs(x - y)-epsilon)))

def radicals(t):
    r1 = 0.8
    r2 = 0.8 
    r3 = 0.8
    return np.array([r1, r2, r3])

def simuler(n):
    X = np.random.normal(loc=0.5, scale=0.1, size=n)
    historique = np.empty((0, n))
    historique = np.vstack((historique, X))
    for t in range(1000):
        R = np.empty((0,3))
        #R = radicals(t)
        new_X = X.copy()
        for i in range(n):
            A = 0
            B = 0
            for j in range(n):
                if j != i:
                    influence = phi(X[j], X[i])
                    A += (X[j] - X[i]) * influence
                    B += influence
            for k in range(len(R)):
                influence = phi(R[k], X[i])
                A += (R[k] - X[i]) * influence
                B += influence
            new_X[i] += alpha * A / B
        X = new_X
        historique = np.vstack((historique, X))
    return historique

def plotter(n):
    historique = simuler(n)
    for agent in range(n):
        plt.plot(historique[:, agent], alpha=1, linewidth=1)
    plt.show()

plotter(50)

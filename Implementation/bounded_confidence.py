import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binned_statistic

alpha = 0.1

def simuler_bounded_confidence(n,epsilon_right,epsilon_left):
    opinions = np.linspace(0,1,n)
    historique=np.empty((0,n))
    for t in range(1000):
        updated_opinions = np.zeros_like(opinions)
        for agent in range(n):
            neighbors = opinions[(- epsilon_left <= opinions - opinions[agent] ) & (opinions - opinions[agent] <=  epsilon_right)]
            updated_opinions[agent] = alpha * np.mean(neighbors) + (1-alpha)*opinions[agent]
        opinions = updated_opinions
        historique = np.vstack((historique, opinions))
    return historique

def plotter(n,epsilon_right,epsilon_left):
    historique =simuler_bounded_confidence(n,epsilon_right,epsilon_left)
    for agent in range(n):
            plt.plot(historique[:, agent],alpha=1, linewidth=1)
    plt.show()

plotter(50,0.1,0.1)

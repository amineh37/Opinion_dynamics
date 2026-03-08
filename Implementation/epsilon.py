import numpy as np
import matplotlib.pyplot as plt

sigma=10
min_eps=0.1
max_eps=0.2

def epsilon(i, sigma, n_agents, min_eps, max_eps):
    center = (n_agents - 1) / 2 
    raw = np.exp(-(i - center)**2 / (2 * sigma**2))
    return min_eps + (max_eps - min_eps) * raw  

def simuler_bounded_confidence(n):
    opinions = np.linspace(0,1,n)
    historique=opinions
    t=0
    convergence=False
    while convergence==False and t<1000:
        t+=1
        updated_opinions = np.zeros_like(opinions)
        for agent in range(n):
            neighbors = opinions[(abs(opinions - opinions[agent]) <=  epsilon(agent,sigma,n,min_eps,max_eps))]
            updated_opinions[agent] = np.mean(neighbors)
        max_change = np.max(np.abs(opinions - updated_opinions))
        if max_change < 10**-5:
            convergence = True
        opinions = updated_opinions
        historique = np.vstack((historique, opinions))   
    return historique


def plotter(n):
    historique =simuler_bounded_confidence(n)
    for agent in range(n):
            plt.plot(historique[:, agent],alpha=1, linewidth=1)
    plt.show()

plotter(50)

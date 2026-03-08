import numpy as np
import matplotlib.pyplot as plt

def f(t):
     return 1-1/(t**0.5+1)


def simuler_bounded_confidence(n, epsilon_right, epsilon_left):
    opinions = np.linspace(0, 1, n)
    historique = [opinions.copy()]
    t=0
    convergence=False
    while convergence==False and t<1000:
        t+=1
        updated_opinions = np.zeros_like(opinions)
        for agent in range(n):
            diffs = opinions - opinions[agent]
            mask = (-epsilon_left <= diffs) & (diffs <= epsilon_right)
            mask[agent] = False 
            if np.any(mask): 
                updated_opinions[agent] = (1-f(t))*np.mean(opinions[mask]) + f(t)*opinions[agent]
            else: 
                updated_opinions[agent] = opinions[agent]
        max_change = np.max(np.abs(opinions - updated_opinions))
        if max_change < 10**-5:
            convergence = True        
        opinions = updated_opinions
        historique.append(opinions.copy())
    
    return np.array(historique)

def plotter(n,epsilon_right,epsilon_left):
    historique =simuler_bounded_confidence(n,epsilon_right,epsilon_left)
    for agent in range(n):
            plt.plot(historique[:, agent],alpha=1, linewidth=1)
    plt.show()

plotter(50,0.2,0.2)

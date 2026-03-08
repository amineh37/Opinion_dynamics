import numpy as np
import matplotlib.pyplot as plt

def BC_avec_radicals(n_normals,n_radicals,epsilon,R):
    n=n_normals+n_radicals
    new_radicals=0
    opinions = np.append(np.linspace(0, 1, n_normals), np.full(n_radicals,R))
    alpha=[0 for _ in range(n_normals)]+[1 for _ in range(n_radicals)]
    historique=opinions
    t=0
    convergence=False
    while convergence==False and t<8:
        if t == 2:
             updated_opinions[len(opinions)-n_radicals:len(opinions)] = np.full(n_radicals,0.75)
        t+=1
        updated_opinions = np.zeros_like(opinions)
        for agent in range(n):
            neighbors = opinions[(abs(opinions - opinions[agent]) <=  epsilon)]
            updated_opinions[agent] = np.mean(neighbors)*(1-alpha[agent])+alpha[agent]*opinions[agent]
        max_change = np.max(np.abs(opinions - updated_opinions))
        if max_change < 10**-5:
            convergence = True
        opinions = updated_opinions
        historique = np.vstack((historique, opinions))  
    for agent in range(n_normals):
          if abs(R-opinions[agent]) <= 10**-3:
                new_radicals+=1
    print(new_radicals)
    return historique

def plotter(n_normals,n_radicals,epsilon,R):
    historique = BC_avec_radicals(n_normals,n_radicals,epsilon,R)
    for agent in range(n_normals+n_radicals):
            plt.plot(historique[:, agent],alpha=1, linewidth=1)
    plt.show()

plotter(50,10,0.1,0.8)
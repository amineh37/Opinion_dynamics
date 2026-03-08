import numpy as np
import matplotlib.pyplot as plt
from ot import emd

current_dist = np.array([10662, 3130, 4538, 4725, 12273, 8162, 9681, 11497, 6882, 21454])
target_dist = np.array([5000, 1000, 3000, 3500, 11000, 9000, 11000, 14000, 9500, 25904])

opinion_bins = np.arange(1, 11) 

mu = current_dist / current_dist.sum() 
nu = target_dist / target_dist.sum()    

C = (np.abs(opinion_bins[:, None] - opinion_bins[None, :])**2)/2

T = emd(mu, nu, C)

print(np.floor(T*93004))

plt.figure(figsize=(12, 6))
plt.bar(opinion_bins , current_dist, width=0.8, alpha=0.7)
plt.bar(opinion_bins , target_dist, width=0.4, alpha=0.7)
plt.show()
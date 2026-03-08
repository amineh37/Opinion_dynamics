import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution
from scipy.stats import wasserstein_distance  # Changement ici
from scipy.stats import norm

# === Paramètres ===
T = 5
n_normals = 50
n_radicals = 2
bins = 100
opinion_bins = np.arange(1, bins+1)
bin_edges = np.linspace(0, 1, bins+1)
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
beta = 100
alpha = 0.1

# === Distribution cible : gaussienne centrée à droite ===
target_pdf = norm.pdf(bin_centers, loc=0.7, scale=0.1)
target_dist = target_pdf / target_pdf.sum()

def phi(x, beta):
    # Fonction sigmoïde inversée
    return 1 / (1 + np.exp(beta * x))

def simulate_continuous_model(u, n_normals, n_radicals, T, alpha=0.1, beta=10):
    u = np.array(u).reshape((T, n_radicals))
    opinions = np.clip(np.random.normal(loc=0.4, scale=0.1, size=n_normals), 0, 1)

    for t in range(T):
        radicals = u[t]
        total_opinions = np.concatenate([opinions, radicals])
        updated_opinions = np.copy(opinions)

        for i in range(n_normals):
            xi = opinions[i]
            xj = total_opinions
            influence_weights = phi(xi - xj, beta)
            influence_weights[i] = 0  # pas d'auto-influence
            sum_weights = np.sum(influence_weights)
            if sum_weights > 0:
                influence = np.sum(influence_weights * (xj - xi)) / sum_weights
                updated_opinions[i] = xi + alpha * influence
        opinions = np.clip(updated_opinions, 0, 1)
    
    return opinions


# === Fonction coût utilisant la distance de Wasserstein ===
def make_cost_function(n_normals, n_radicals, T, alpha, beta, target_dist):
    def cost(u):
        final_opinions = simulate_continuous_model(u, n_normals, n_radicals, T, alpha=alpha, beta=beta)
        return wasserstein_distance(
            final_opinions,
            np.random.choice(bin_centers, size=len(final_opinions), p=target_dist)
        )
    return cost


# === Optimisation ===
cost_fn = make_cost_function(n_normals, n_radicals, T, alpha, beta, target_dist)
bounds = [(0, 1)] * (T * n_radicals)

result = differential_evolution(cost_fn, bounds, maxiter=40, disp=True)
u_optimal = result.x.reshape((T, n_radicals))
final_opinions = simulate_continuous_model(result.x, n_normals, n_radicals, T, alpha , beta)

# === Affichage ===
plt.figure(figsize=(10, 5))

# Histogramme de la distribution finale (normalisé)
plt.hist(final_opinions, bins=bin_edges, alpha=0.6, label="Finale", edgecolor='black', density=True)

# Histogramme de la distribution cible
target_sample = np.random.choice(bin_centers, size=10000, p=target_dist)  # échantillon synthétique
plt.hist(target_sample, bins=bin_edges, alpha=0.6, label="Cible", edgecolor='black', density=True)

plt.legend()
plt.title("Histogrammes comparés : distribution finale vs cible")
plt.xlabel("Opinion")
plt.ylabel("Densité")
plt.xlim(0, 1)
plt.grid(True)
plt.show()

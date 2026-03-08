import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution

# === Paramètres ===
T = 5
n_normals = 50
n_radicals = 2
bins = 100
bin_edges = np.linspace(0, 1, bins+1)
beta = 100
alpha = 0.1

def phi(x, beta):
    # Fonction sigmoïde inversée
    return 1 / (1 + np.exp(beta * x))

def simulate_continuous_model(u, initial_opinions, n_radicals, T, alpha=0.1, beta=10):
    u = np.array(u).reshape((T, n_radicals))
    opinions = np.copy(initial_opinions) # Utiliser une copie pour ne pas modifier l'original
    n_normals = len(opinions)

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

# === Nouvelle fonction objectif : maximiser la somme des opinions ===
def make_objective_function(initial_opinions, n_radicals, T, alpha, beta):
    def objective(u):
        # Simuler pour obtenir les opinions finales
        final_opinions = simulate_continuous_model(u, initial_opinions, n_radicals, T, alpha=alpha, beta=beta)
        # On veut maximiser la somme, donc on minimise son opposé
        return -np.sum(final_opinions)
    return objective

# === Génération des opinions initiales (une seule fois) ===
initial_opinions = np.clip(np.random.normal(loc=0.4, scale=0.1, size=n_normals), 0, 1)

# === Optimisation ===
objective_fn = make_objective_function(initial_opinions, n_radicals, T, alpha, beta)
bounds = [(0, 1)] * (T * n_radicals)

result = differential_evolution(objective_fn, bounds, maxiter=40, disp=True, workers=-1) # workers=-1 pour paralléliser
u_optimal = result.x.reshape((T, n_radicals))
final_opinions = simulate_continuous_model(result.x, initial_opinions, n_radicals, T, alpha , beta)

# === Affichage ===
plt.figure(figsize=(10, 6))
plt.hist(initial_opinions, bins=bin_edges, alpha=0.6, label="Opinions Initiales", density=True, color='gray')
plt.hist(final_opinions, bins=bin_edges, alpha=0.8, label="Opinions Finales (Maximizées)", density=True, color='red')
plt.axvline(np.mean(initial_opinions), color='gray', linestyle='--', label=f'Moyenne Initiale: {np.mean(initial_opinions):.2f}')
plt.axvline(np.mean(final_opinions), color='red', linestyle='--', label=f'Moyenne Finale: {np.mean(final_opinions):.2f}')

plt.legend()
plt.title("Distribution des opinions avant et après maximisation")
plt.xlabel("Opinion (0 = gauche, 1 = droite)")
plt.ylabel("Densité")
plt.xlim(0, 1)
plt.grid(True, linestyle=':', alpha=0.6)
plt.show()
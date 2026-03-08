import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

def BC_avec_radicals_sans_historique(n_normals, n_radicals, epsilon,R):
    n = n_normals + n_radicals
    new_radicals = 0
    opinions = np.append(np.linspace(0, 1, n_normals), np.full(n_radicals,R))
    alpha = [0 for _ in range(n_normals)] + [1 for _ in range(n_radicals)]
    converged = False
    t = 0
    while not converged and t < 1000:
        t += 1
        updated_opinions = np.zeros_like(opinions)
        for agent in range(n):
            neighbors = opinions[(abs(opinions - opinions[agent]) <= epsilon)]
            updated_opinions[agent] = np.mean(neighbors)*(1-alpha[agent]) + alpha[agent]*opinions[agent]
        if np.max(np.abs(opinions - updated_opinions)) < 1e-5:
            converged = True
        opinions = updated_opinions
    for agent in range(n_normals):
        if abs(R - opinions[agent]) <= 1e-3:
            new_radicals += 1
    return new_radicals


def wavelength_to_rgb(wavelength, gamma=0.8):
    """Convert wavelength in nm to RGB color (approximate visible spectrum)"""
    wavelength = float(wavelength)
    
    # Color mapping (approximate visible spectrum)
    if wavelength < 380: wavelength = 380
    if wavelength > 750: wavelength = 750
    
    if wavelength < 440:
        attenuation = 0.3 + 0.7 * (wavelength - 380) / (440 - 380)
        R = ((-(wavelength - 440) / (440 - 380)) * attenuation) ** gamma
        G = 0.0
        B = (1.0 * attenuation) ** gamma
    elif wavelength < 490:
        R = 0.0
        G = ((wavelength - 440) / (490 - 440)) ** gamma
        B = 1.0
    elif wavelength < 510:
        R = 0.0
        G = 1.0
        B = (-(wavelength - 510) / (510 - 490)) ** gamma
    elif wavelength < 580:
        R = ((wavelength - 510) / (580 - 510)) ** gamma
        G = 1.0
        B = 0.0
    elif wavelength < 645:
        R = 1.0
        G = (-(wavelength - 645) / (645 - 580)) ** gamma
        B = 0.0
    else:
        attenuation = 0.3 + 0.7 * (750 - wavelength) / (750 - 645)
        R = (1.0 * attenuation) ** gamma
        G = 0.0
        B = 0.0
    
    return (R, G, B)

wavelengths = np.linspace(380, 750, 256)
spectrum_colors = [wavelength_to_rgb(w) for w in wavelengths]
spectral_cmap = LinearSegmentedColormap.from_list('spectral', spectrum_colors, N=256)


def grid(n_normals,R):
    epsilons = np.linspace(0.01, 0.5, 50) 
    n_radicals_range = np.arange(1, 51, 1)


    grid = np.zeros((len(n_radicals_range), len(epsilons)))
    for i, n_rad in enumerate(n_radicals_range):
        for j, eps in enumerate(epsilons):
            grid[i, j] = BC_avec_radicals_sans_historique(n_normals, n_rad, eps,R)

    plt.figure(figsize=(10, 8))
    im = plt.imshow(grid, 
                    extent=[epsilons.min(), epsilons.max(), 
                        n_radicals_range.min(), n_radicals_range.max()],
                    aspect='auto', 
                    cmap=spectral_cmap, 
                    origin='lower')
    plt.show()

grid(50,1)
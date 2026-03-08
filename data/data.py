import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

fichiers = [
    ("WV1_Data.csv", "V66"),
    ("WV3_Data.csv", "V66"),
    ("WV4_Data.csv", "V82"),
    ("WV5_Data.csv", "V46"),
    ("WV6_Data.csv", "V55"),
    ("WV7_Data.csv", "Q48"),
]

annees = [1981, 1990, 1995, 2000, 2005, 2010]

fig, axes = plt.subplots(2, 3, figsize=(18, 8), sharey=True)

for idx, ((fichier, colonne), annee) in enumerate(zip(fichiers, annees)):
    sep = "," if "WV7" in fichier else ";"
    df = pd.read_csv(fichier, sep=sep, low_memory=False, on_bad_lines='skip')
    df.columns = df.columns.str.strip()
    
    opinions = pd.to_numeric(df[colonne], errors='coerce').dropna()
    opinions = (opinions - opinions.min()) / (opinions.max() - opinions.min())  

    # Histogramme avec comptage absolu
    hist, bins = np.histogram(opinions, bins=10, range=(0, 1), density=False)
    centers = 0.5 * (bins[:-1] + bins[1:])

    ax = axes[idx // 3, idx % 3]
    ax.bar(centers, hist, width=0.08, edgecolor='black')
    ax.set_title(f"Année {annee}")
    ax.set_xlabel("Opinion")
    if idx % 3 == 0:
        ax.set_ylabel("Nombre d'agents")

fig.suptitle("Histogrammes des opinions par année (sans 1986)", fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()


import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from hmmlearn import hmm

# 1. Téléchargement des données Apple
data = yf.download("AAPL", start="2022-01-01", end="2023-12-31")

# 2. Calcul des rendements logarithmiques (plus stables que les différences brutes)
data['Returns'] = np.log(data['Close'] / data['Close'].shift(1))
data = data.dropna()

# 3. Paramétrage et entraînement du HMM (GaussianHMM pour des données continues)
n_states = 3  # 3 états : haussier, baissier, neutre
model = hmm.GaussianHMM(
    n_components=n_states,
    covariance_type="diag",  # Variance différente pour chaque état
    n_iter=1000,
    random_state=42
)

# Reshape des rendements pour le modèle
X = data[['Returns']].values

# Entraînement
model.fit(X)

# Prédiction des états cachés
hidden_states = model.predict(X)
data['Hidden_State'] = hidden_states

# 4. Interprétation des états (tri par rendement moyen)
state_stats = data.groupby('Hidden_State')['Returns'].mean().sort_values()
state_order = state_stats.index

# Associer chaque état à un label et une couleur en fonction du rendement moyen
state_info = {
    state_order[0]: {"label": "↓ Baissier", "color": "#ff9999"},  # Rouge pour le rendement le plus bas
    state_order[1]: {"label": "= Neutre", "color": "#99ccff"},    # Bleu pour le rendement intermédiaire
    state_order[2]: {"label": "↑ Hausier", "color": "#aaffaa"}    # Vert pour le rendement le plus haut
}

# 5. Visualisation
plt.figure(figsize=(14, 7))

# Tracé des prix colorés par état
for i in range(len(data) - 1):
    state = hidden_states[i]
    plt.plot(data.index[i:i+2], data['Close'].iloc[i:i+2], 
             color=state_info[state]["color"], linewidth=2)

# Légende
patches = [
    mpatches.Patch(color=state_info[state]["color"], label=state_info[state]["label"]) 
    for state in state_order
]
plt.legend(handles=patches, title="États du Marché")

plt.title("États cachés du marché (Apple) prédits par HMM")
plt.xlabel("Date")
plt.ylabel("Prix de Clôture ($)")
plt.grid(True)
plt.tight_layout()
plt.show()

# 6. Statistiques des états (optionnel)
print("\nRendements moyens par état:")
print(data.groupby('Hidden_State')['Returns'].mean())

print("\nMatrice de transition entre états:")
print(model.transmat_)
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 1. États possibles
etats = ["Soleil", "Pluie"]

# 2. Matrice de transition (ordre : [Soleil, Pluie])
# ex : P(Soleil→Soleil) = 0.8, P(Soleil→Pluie) = 0.2, etc.
transition_matrix = np.array([
    [0.8, 0.2],  # depuis Soleil
    [0.4, 0.6]   # depuis Pluie
])

# 3. Simulation d'une séquence
def simulate_markov_chain(start_state, n_steps):
    current_state = etats.index(start_state)
    sequence = [start_state]

    for _ in range(n_steps - 1):
        current_state = np.random.choice([0, 1], p=transition_matrix[current_state])
        sequence.append(etats[current_state])

    return sequence

# Génère une séquence de 30 jours
sequence = simulate_markov_chain(start_state="Soleil", n_steps=30)

# Affichage
print("Séquence simulée :")
print(sequence)

# 4. Visualisation (comptage de chaque état par position)
plt.figure(figsize=(12, 2))
plt.plot(sequence, marker="o", linestyle="dashed")
plt.title("Simulation d'une chaîne de Markov météo (Soleil / Pluie)")
plt.xticks(rotation=45)
plt.grid(True)
plt.show()

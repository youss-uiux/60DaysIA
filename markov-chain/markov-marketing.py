import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

# États du parcours
etats = ["Accueil", "Produit", "Panier", "Achat", "Quitte"]
etat_final = ["Achat", "Quitte"]

# Matrice de transition (ligne = état actuel, colonnes = états suivants)
transition_matrix = np.array([
    [0.1, 0.6, 0.0, 0.0, 0.3],  # Accueil
    [0.0, 0.2, 0.5, 0.0, 0.3],  # Produit
    [0.0, 0.0, 0.2, 0.6, 0.2],  # Panier
    [0.0, 0.0, 0.0, 1.0, 0.0],  # Achat (état absorbant)
    [0.0, 0.0, 0.0, 0.0, 1.0]   # Quitte (état absorbant)
])

# Simulation d’un seul utilisateur
def simuler_utilisateur(start_state="Accueil"):
    parcours = [start_state]
    current_state = etats.index(start_state)
    
    while etats[current_state] not in etat_final:
        current_state = np.random.choice(len(etats), p=transition_matrix[current_state])
        parcours.append(etats[current_state])
    
    return parcours

# Simulation de N utilisateurs
def simuler_plusieurs_utilisateurs(N=1000):
    resultats = {"Achat": 0, "Quitte": 0}
    parcours_exemples = []
    
    for _ in range(N):
        chemin = simuler_utilisateur()
        resultats[chemin[-1]] += 1
        if len(parcours_exemples) < 5:
            parcours_exemples.append(chemin)
    
    return resultats, parcours_exemples

# Lance la simulation
resultats, exemples = simuler_plusieurs_utilisateurs(1000)

# Affiche les résultats
print("📊 Résultats de la simulation sur 1000 utilisateurs :")
print(f"- Ont acheté : {resultats['Achat']} ({resultats['Achat']/10:.1f}%)")
print(f"- Ont quitté : {resultats['Quitte']} ({resultats['Quitte']/10:.1f}%)\n")

# Affiche quelques exemples de parcours
print("🧾 Exemples de parcours utilisateur :")
for parcours in exemples:
    print(" → ".join(parcours))

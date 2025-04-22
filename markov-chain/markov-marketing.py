import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import networkx as nx

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


# Création du graphe dirigé
G = nx.DiGraph()

# Ajout des arcs avec poids
for i in range(len(etats)):
    for j in range(len(etats)):
        prob = transition_matrix[i, j]
        if prob > 0:
            G.add_edge(etats[i], etats[j], weight=prob)

# Position des nœuds pour un affichage clair
pos = nx.spring_layout(G, seed=42)  # layout aléatoire mais fixe

# Dessin des nœuds et des flèches
plt.figure(figsize=(10, 7))
nx.draw(G, pos, with_labels=True, node_size=2500, node_color="lightblue", font_size=10, font_weight="bold", arrowsize=20)

# Affichage des poids (probabilités)
edge_labels = {(u, v): f"{d['weight']:.2f}" for u, v, d in G.edges(data=True)}
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=9)

plt.title("Chaîne de Markov – Parcours utilisateur e-commerce")
plt.axis("off")
plt.show()

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


# Génération du diagramme de flux
plt.figure(figsize=(len(parcours)*2.5, 2))

# Placement horizontal des étapes
for i, etape in enumerate(parcours):
    plt.text(i, 0, etape, fontsize=12, ha='center', va='center',
             bbox=dict(boxstyle="round,pad=0.4", facecolor='lightblue', edgecolor='black'))

    if i < len(parcours) - 1:
        plt.arrow(i + 0.4, 0, 0.2, 0, head_width=0.05, head_length=0.1, fc='gray', ec='gray')

plt.axis("off")
plt.title("Parcours utilisateur simulé", fontsize=14)
plt.tight_layout()
plt.show()


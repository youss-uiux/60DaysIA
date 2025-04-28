import networkx as nx
import matplotlib.pyplot as plt

# Créer le graphe du réseau bayésien
G = nx.DiGraph([
    ("Taux", "VIX"),
    ("Taux", "SP500"),
    ("VIX", "Cacao"),
    ("SP500", "Cacao"),
    ("EURUSD", "Cacao"),
    ("USDX", "Cacao"),
    ("Oil", "Cacao")
])

# Définir la disposition
pos = {
    "Taux": (0, 2),
    "VIX": (-1, 1),
    "SP500": (1, 1),
    "EURUSD": (-2, 0),
    "USDX": (0, 0),
    "Oil": (2, 0),
    "Cacao": (0, -1)
}

# Dessiner le graphe
plt.figure(figsize=(8, 6))
nx.draw(
    G, pos, 
    with_labels=True, 
    node_color="#1E3A8A",  # Bleu tech
    node_size=3000, 
    font_size=12, 
    font_weight="bold", 
    font_color="white", 
    edge_color="#4A7043",  # Vert cacao
    arrowsize=20
)

# Ajouter un titre
plt.title("Réseau Bayésien : Prédiction du Prix du Cacao\nJour 20/60", fontsize=14, fontweight="bold", color="#1E3A8A")

# Sauvegarder l'image
plt.savefig("bayesian_network_cacao.png", dpi=300, bbox_inches="tight")
print("Image sauvegardée sous 'bayesian_network_cacao.png'")
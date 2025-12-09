import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

# -----------------------------
# 1. Chargement et préparation des données Iris
# -----------------------------
iris = load_iris()
X = iris.data.astype(np.float32)  # 4 caractéristiques
y = iris.target.astype(np.int64)  # 0, 1 ou 2

# Normalisation
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Conversion en tenseurs
X_tensor = torch.from_numpy(X)
y_tensor = torch.from_numpy(y)


# -----------------------------
# 2. Modèle ANN pour classification multi-classes
# -----------------------------
class IrisClassifier(nn.Module):
    def __init__(self, input_size=4, num_classes=3):
        super(IrisClassifier, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 16),
            nn.ReLU(),
            nn.Linear(16, 12),
            nn.ReLU(),
            nn.Linear(12, num_classes)  # 3 neurones de sortie (pas de softmax ici)
        )

    def forward(self, x):
        return self.network(x)


model = IrisClassifier()

# -----------------------------
# 3. Loss et optimiseur adaptés à la multi-class
# -----------------------------
criterion = nn.CrossEntropyLoss()  # Combine softmax + NLLLoss
optimizer = optim.Adam(model.parameters(), lr=0.01)

# -----------------------------
# 4. Entraînement
# -----------------------------
model.train()
for epoch in range(300):
    optimizer.zero_grad()

    outputs = model(X_tensor)  # shape: (150, 3)
    loss = criterion(outputs, y_tensor)

    loss.backward()
    optimizer.step()

    if (epoch + 1) % 50 == 0:
        # Calcul de l'accuracy
        _, predicted = torch.max(outputs, 1)
        accuracy = (predicted == y_tensor).sum().item() / len(y_tensor)
        print(f"Epoch {epoch + 1}/300 | Loss: {loss.item():.4f} | Accuracy: {accuracy * 100:.2f}%")

# -----------------------------
# 5. Prédiction sur de nouvelles données
# -----------------------------
model.eval()
with torch.no_grad():
    # Exemple : une nouvelle fleur avec ces mesures
    nouvelle_fleur = np.array([[5.1, 3.5, 1.4, 0.2]], dtype=np.float32)
    nouvelle_fleur_scaled = scaler.transform(nouvelle_fleur)
    nouvelle_tensor = torch.from_numpy(nouvelle_fleur_scaled)

    sortie = model(nouvelle_tensor)
    _, prediction = torch.max(sortie, 1)

    noms_classes = ["Setosa", "Versicolor", "Virginica"]
    classe_predite = noms_classes[prediction.item()]

    print(f"\nPrédiction pour la nouvelle fleur : {classe_predite}")
    print(f"Probabilités : {torch.softmax(sortie, dim=1).numpy()[0]}")
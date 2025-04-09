import xml.etree.ElementTree as ET
import numpy as np
import matplotlib.pyplot as plt

# === Charger et parser le fichier XML ===
tree = ET.parse("data2.xml")
root = tree.getroot()

annees = []
valeurs = []

for record in root.findall(".//record"):
    country = record.find("./field[@name='Country or Area']")
    year = record.find("./field[@name='Year']")
    value = record.find("./field[@name='Value']")

    if country is not None and country.text == "Niger" and value is not None and value.text:
        try:
            annee = int(year.text)
            # Remplacer la virgule par un point si elle existe
            val = float(value.text.replace(',', '.'))
            annees.append(annee)
            valeurs.append(val)
        except ValueError:
            continue  # ignorer les valeurs non convertibles

# === Convertir en tableau numpy trié ===
x = np.array(annees)
y = np.array(valeurs)

# Trier les données si nécessaire
x, y = zip(*sorted(zip(x, y)))
x = np.array(x)
y = np.array(y)

# === Moindres carrés ===
x_mean = x.mean()
y_mean = y.mean()
a_mco = np.sum((x - x_mean)*(y - y_mean)) / np.sum((x - x_mean)**2)
b_mco = y_mean - a_mco * x_mean
y_pred_mco = a_mco * x + b_mco

# === Descente de gradient ===
x_norm = (x - x.mean()) / x.std()
y_norm = (y - y.mean()) / y.std()

a_grad, b_grad = 0, 0
alpha = 0.01
epochs = 1000
n = len(x_norm)

for _ in range(epochs):
    y_pred = a_grad * x_norm + b_grad
    error = y_pred - y_norm
    grad_a = (2/n) * np.sum(error * x_norm)
    grad_b = (2/n) * np.sum(error)
    a_grad -= alpha * grad_a
    b_grad -= alpha * grad_b

# Dénormaliser la droite trouvée
a_final = (y.std() / x.std()) * a_grad
b_final = y.mean() + y.std() * b_grad - a_final * x.mean()
y_pred_grad = a_final * x + b_final

# === Affichage ===
plt.figure(figsize=(10,6))
plt.scatter(x, y, label="Données réelles")
plt.plot(x, y_pred_mco, color="red", label=f"MCO : y = {a_mco:.2f}x + {b_mco:.2f}")
plt.plot(x, y_pred_grad, color="green", linestyle='--', label=f"Gradient : y = {a_final:.2f}x + {b_final:.2f}")
plt.xlabel("Année")
plt.ylabel("Accès à l’électricité (%)")
plt.title("Régression linéaire - Accès à l’électricité au Niger")
plt.legend()
plt.grid()
plt.show()

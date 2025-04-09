import xml.etree.ElementTree as ET
import numpy as np
import matplotlib.pyplot as plt

# ========== 1. Charger et parser le fichier XML ==========
tree = ET.parse("data2.xml")  # remplace par ton vrai nom de fichier
root = tree.getroot()

annees = []
populations = []

for record in root.findall(".//record"):
    country_field = record.find("./field[@name='Country or Area']")
    year_field = record.find("./field[@name='Year']")
    value_field = record.find("./field[@name='Value']")

    if (
        country_field is not None and country_field.text == "Niger"
        and year_field is not None and value_field is not None
    ):
        year = int(year_field.text)
        value = value_field.text
        if value:
            annees.append(year)
            populations.append(int(value))

# ========== 2. Convertir et normaliser les données ==========
x = np.array(annees)
y = np.array(populations)

# Tri par année (au cas où)
x, y = zip(*sorted(zip(x, y)))
x = np.array(x)
y = np.array(y)

# Normalisation
x_norm = (x - x.mean()) / x.std()
y_norm = (y - y.mean()) / y.std()

# ========== 3. Méthode des moindres carrés ==========
x_mean = x.mean()
y_mean = y.mean()

a_mco = np.sum((x - x_mean) * (y - y_mean)) / np.sum((x - x_mean)**2)
b_mco = y_mean - a_mco * x_mean

y_pred_mco = a_mco * x + b_mco

# ========== 4. Descente de gradient ==========
a_grad, b_grad = 0, 0
alpha = 0.01
epochs = 1000
n = len(x_norm)
loss_history = []

for _ in range(epochs):
    y_pred = a_grad * x_norm + b_grad
    error = y_pred - y_norm
    loss = (error**2).mean()
    loss_history.append(loss)

    grad_a = (2/n) * np.sum(error * x_norm)
    grad_b = (2/n) * np.sum(error)

    a_grad -= alpha * grad_a
    b_grad -= alpha * grad_b

# Dé-normalisation des prédictions
x_renorm = x_norm * x.std() + x.mean()
y_pred_grad_norm = a_grad * x_norm + b_grad
y_pred_grad = y_pred_grad_norm * y.std() + y.mean()

# Recalcul des coefficients dans l'échelle originale
a_final = (y.std() / x.std()) * a_grad
b_final = y.mean() + y.std() * b_grad - a_final * x.mean()

# ========== 5. Visualisation ==========
plt.figure(figsize=(10, 6))
plt.scatter(x, y, label="Données réelles")
plt.plot(x, y_pred_mco, color="red", label=f"MCO : y = {a_mco:.2f}x + {b_mco:.2f}")
plt.plot(x, y_pred_grad, color="green", linestyle="--", label=f"Gradient : y = {a_final:.2f}x + {b_final:.2f}")
plt.xlabel("Année")
plt.ylabel("Population")
plt.title("Régression linéaire sur les données de la population du Niger")
plt.legend()
plt.grid()
plt.show()

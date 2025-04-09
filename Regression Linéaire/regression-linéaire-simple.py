import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import numpy as np

# Charger le fichier XML
tree = ET.parse("data.xml")
root = tree.getroot()

# Extraire les données du Niger
annees = []
populations = []

for record in root.findall(".//record"):
    country = record.find("./field[@name='Country or Area']").text
    if country == "Niger":
        year = int(record.find("./field[@name='Year']").text)
        value = record.find("./field[@name='Value']").text
        if value:  # s'assurer qu'il y a une valeur
            annees.append(year)
            populations.append(int(value))

# Convertir en tableaux NumPy
x = np.array(annees)
y = np.array(populations)
# Calculer les coefficients de la régression linéaire
x_mean=x.mean()
y_mean=y.mean()

# Calculs
a = np.sum((x - x_mean) * (y - y_mean)) / np.sum((x - x_mean)**2)
b = y_mean - a * x_mean

# Prédictions
y_pred = a * x + b
# Affichage
plt.scatter(x, y)
plt.plot(x, y_pred, color='red', label=f"Régression : y = {a:.2f}x + {b:.2f}")
plt.legend()
plt.grid()
plt.title("Moindres carrés")
plt.show()


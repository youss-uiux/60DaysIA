import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
from sklearn.linear_model import BayesianRidge
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import pandas as pd
from functools import reduce
import bambi as bmb
import arviz as az
import numpy as np


def extract_indicator_from_xml(file_path, indicator_label, country="Niger"):
    tree = ET.parse(file_path)
    root = tree.getroot()

    data = []
    for record in root.findall(".//record"):
        country_name = record.find("./field[@name='Country or Area']")
        year = record.find("./field[@name='Year']")
        value = record.find("./field[@name='Value']")
        
        if country_name is not None and year is not None and value is not None:
            if country_name.text == country and value.text and year.text:
                try:
                    data.append({
                        "Année": int(year.text),
                        indicator_label: float(value.text.replace(",", ""))
                    })
                except:
                    continue

    return pd.DataFrame(data)

# === Fichiers XML à fusionner ===
fichiers = {
    "Acces a lelectricite.xml": "Accès à l’électricité",
    "Chomage.xml": "Chômage",
    "utilisateur internet.xml": "Utilisateurs d’Internet",
    "investissement étranger.xml":"Investissement étranger direct (% du PIB)",
    "envoi.xml": "Envois de fonds (% du PIB)",
}

dfs = [extract_indicator_from_xml(fichier, label) for fichier, label in fichiers.items()]
df_final = reduce(lambda left, right: pd.merge(left, right, on="Année", how="outer"), dfs)

# 1. Charge tes données
df = df_final.copy()
df = df.dropna()
df = df.drop(columns=["Année"])  # Si Année n’est pas utile

# 2. Définis la cible (Y) et les variables explicatives (X)
X = df.drop(columns=["Accès à l’électricité"])  # ou la variable que tu veux prédire
y = df["Accès à l’électricité"]

# 3. Split train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 4. Modèle bayésien gaussien
model = BayesianRidge()
model.fit(X_train, y_train)

# 5. Prédictions et évaluation
y_mean, y_std = model.predict(X_test, return_std=True)


plt.figure(figsize=(10, 6))
plt.plot(y_test.values, label="Valeur réelle", marker='o')
plt.plot(y_mean, label="Prédiction", marker='x')
plt.fill_between(
    np.arange(len(y_mean)),
    y_mean - 1.96 * y_std,
    y_mean + 1.96 * y_std,
    color='gray',
    alpha=0.3,
    label="Intervalle de confiance 95%"
)
plt.title("Régression bayésienne gaussienne avec intervalle de confiance")
plt.legend()
plt.grid(True)
plt.show()

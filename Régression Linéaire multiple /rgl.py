import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import statsmodels.api as sm
import pandas as pd
from functools import reduce

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
    "espérence de vie.xml": "Espérance de vie",
    "Acces a lelectricite.xml": "Accès à l’électricité",
    "Chomage.xml": "Chômage",
    "croissance du pib.xml": "Croissance du PIB",
    "envoi.xml": "Envois de fonds (% du PIB)",
    "Spi.xml": "SPI (score statistique)",
    "utilisateur internet.xml": "Utilisateurs d’Internet"
}

dfs = [extract_indicator_from_xml(fichier, label) for fichier, label in fichiers.items()]
df_final = reduce(lambda left, right: pd.merge(left, right, on="Année", how="outer"), dfs)

print(df_final.head())

# === Traitement pour interpolation ===
# Revenir à une structure classique
df_final.reset_index(inplace=True)

df_final = df_final.sort_values("Année")
df_final.set_index("Année", inplace=True)

# Convertir les colonnes en numériques
for col in df_final.columns:
    df_final[col] = pd.to_numeric(df_final[col], errors='coerce')

# Interpolation + forward fill + backward fill
df_final = df_final.interpolate(method='linear')  # entre deux années connues
df_final = df_final.fillna(method='ffill')        # valeurs précédentes
df_final = df_final.fillna(method='bfill')        # valeurs suivantes
df_final = df_final.reset_index()

# === Affichage / export ===
print(df_final.head())

y = df_final["Accès à l’électricité"]

# === Définir les variables explicatives ===
X = df_final.drop(columns=["Année", "Accès à l’électricité"])  # On enlève aussi "Année"

# === Ajouter une constante pour l'interception (β₀) ===
X = sm.add_constant(X)

# === Créer et entraîner le modèle ===
model = sm.OLS(y, X).fit()

# === Résumé des résultats ===
print(model.summary())

# Prédictions du modèle
y_pred = model.predict(X)
r_squared = model.rsquared * 100  # pourcentage

residus = y - y_pred

plt.figure(figsize=(10, 5))
plt.bar(df_final["Année"], residus, color='purple')
plt.axhline(0, color='red', linestyle='--')
plt.title("Résidus du modèle : erreurs entre réel et prédit")
plt.xlabel("Année")
plt.ylabel("Erreur (résidu)")
plt.grid(True)
plt.tight_layout()
plt.show()

#plt.figure(figsize=(8, 5))
#plt.scatter(y_pred, residus, color='orange')
#plt.axhline(0, color='red', linestyle='--')
#plt.title("Résidus en fonction des valeurs prédites")
#plt.xlabel("Valeurs prédites")
#plt.ylabel("Résidus")
#plt.grid(True)
#plt.tight_layout()
#plt.show()

plt.figure(figsize=(8, 5))
plt.hist(residus, bins=15, color='skyblue', edgecolor='black')
plt.title("Distribution des résidus")
plt.xlabel("Résidu")
plt.ylabel("Fréquence")
plt.tight_layout()
plt.show()

#plt.figure(figsize=(10, 6))
#plt.plot(df_final["Année"], y, label="Accès à l'électricité Réel", marker='o', color='blue')
#plt.plot(df_final["Année"], y_pred, label="Accès à l'électricité Prédit", linestyle='--', marker='x', color='orange')

# Titres et axes
#plt.xlabel("Année")
#plt.ylabel("Accès à l’électricité (%)")
#plt.title("Modèle de régression : Accès à l’électricité au Niger")

# Affichage du R² sur le graphique
#plt.text(df_final["Année"].min() + 1, max(y) - 5, f"R² = {r_squared:.2f}%", fontsize=12, color='green')

#plt.legend()
#plt.grid(True)
#plt.tight_layout()
#plt.show()

# df_final.to_csv("donnees_niger_fusionnees_interpolees.csv", index=False)

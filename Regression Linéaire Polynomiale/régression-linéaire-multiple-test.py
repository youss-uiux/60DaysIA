import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
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

# Créer une transformation polynomiale de degré 2 (tu peux mettre 3, 4, etc.)
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X)

# Si tu veux voir les noms des nouvelles colonnes :
poly_feature_names = poly.get_feature_names_out(X.columns)
print(poly_feature_names)

# Tu peux aussi reconstruire un DataFrame lisible si tu veux l’afficher :
import pandas as pd
X_poly_df = pd.DataFrame(X_poly, columns=poly_feature_names)
print(X_poly_df.head())
model = LinearRegression()
model.fit(X_poly, y)
# Prédictions avec le modèle polynomial
y_pred_poly = model.predict(X_poly)

# Affichage
plt.figure(figsize=(10, 6))
plt.plot(df_final["Année"], y, label="Réel", marker='o', color='blue')
plt.plot(df_final["Année"], y_pred_poly, label="Régression polynomiale", linestyle='--', marker='x', color='orange')
# plt.plot(df["Année"], y_pred_linear, label="Régression linéaire", linestyle=':', color='green')  # si besoin

plt.xlabel("Année")
plt.ylabel("Accès à l’électricité (%)")
plt.title("Comparaison : Réel vs Prédit (Régression polynomiale)")
plt.legend()
plt.grid(True)

# Ajouter le R² sur le graphique
r2 = r2_score(y, y_pred_poly)
plt.text(df_final["Année"].min()+1, max(y) - 5, f"R² = {r2*100:.2f}%", fontsize=12, color='green')

plt.tight_layout()
plt.show()

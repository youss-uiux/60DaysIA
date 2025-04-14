import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
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

# Séparation des données
X = df_final.drop(columns=["Année", "Accès à l’électricité"])
y = df_final["Accès à l’électricité"]

# 1. Séparer les données (X et y déjà définis)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 2. Tester différents degrés de polynôme
degrees = range(1, 6)
r2_train_scores = []
r2_test_scores = []

for degree in degrees:
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)

    model = LinearRegression()
    model.fit(X_train_poly, y_train)

    y_pred_train = model.predict(X_train_poly)
    y_pred_test = model.predict(X_test_poly)

    r2_train_scores.append(r2_score(y_train, y_pred_train))
    r2_test_scores.append(r2_score(y_test, y_pred_test))

# 3. Tracer les courbes
plt.figure(figsize=(10, 6))
plt.plot(degrees, r2_train_scores, label="R² Entraînement", marker='o', color='blue')
plt.plot(degrees, r2_test_scores, label="R² Test", marker='x', color='red')
plt.xlabel("Degré du polynôme")
plt.ylabel("Score R²")
plt.title("Courbe d'apprentissage : R² vs Complexité du modèle")
plt.legend()
plt.grid(True)
plt.xticks(degrees)
plt.tight_layout()
plt.show()
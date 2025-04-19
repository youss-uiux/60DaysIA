import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
import pandas as pd
from functools import reduce
import bambi as bmb
import arviz as az

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
    "utilisateur internet.xml": "Utilisateurs d’Internet",
    "investissement étranger.xml":"Investissement étranger direct (% du PIB)",
    "Production d’électricité à partir de sources d’énergie renouvelables.xml":"Production d’électricité à partir de sources d’énergie renouvelables, hors énergie hydroélectrique (% du total)",
    "inflationprix.xml":"Inflation, prix à la consommation (% annuel)",
    "Migration nette.xml":"Migration nette",
    "Croissance de la population (% annuel).xml":"Croissance de la population (% annuel)",

}

dfs = [extract_indicator_from_xml(fichier, label) for fichier, label in fichiers.items()]
df_final = reduce(lambda left, right: pd.merge(left, right, on="Année", how="outer"), dfs)
df_final = df_final.interpolate().fillna(method='ffill').fillna(method='bfill')
df = df_final.drop(columns=["Année"])
# === 1. Transformation polynomiale
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(df.drop(columns=["Inflation, prix à la consommation (% annuel)"]))

# === 2. Nouveau DataFrame avec noms de colonnes
poly_columns = poly.get_feature_names_out(df.drop(columns=["Inflation, prix à la consommation (% annuel)"]).columns)
df_poly = pd.DataFrame(X_poly, columns=poly_columns)

# Ajoute la variable cible (ici : Inflation, à adapter selon ton cas)
df_poly["Inflation"] = df["Inflation, prix à la consommation (% annuel)"]

# === 3. Formule automatique (toutes les variables pour expliquer Inflation)
formula = "Inflation ~ " + " + ".join([f"`{col}`" for col in poly_columns])

# === 4. Modèle bayésien
model = bmb.Model(formula=formula, data=df_poly)
results = model.fit(draws=2000, tune=1000)

# === 5. Résumé
print(az.summary(results))

# === 6. Visualisation
az.plot_trace(results)
plt.tight_layout()
plt.show()
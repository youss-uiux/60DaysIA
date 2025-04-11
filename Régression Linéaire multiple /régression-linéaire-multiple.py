import xml.etree.ElementTree as ET
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
# df_final.to_csv("donnees_niger_fusionnees_interpolees.csv", index=False)

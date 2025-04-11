import xml.etree.ElementTree as ET
import pandas as pd
from functools import reduce

def extract_indicator_from_xml(file_path, indicator_label, country="Niger"):
    """
    Extrait les données pour un indicateur donné depuis un fichier XML de la Banque mondiale.
    """
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

# === Liste des fichiers XML à traiter avec les noms d’indicateurs correspondants ===
fichiers = {
    "espérence de vie.xml": "Espérance de vie",
    "Acces a lelectricite.xml": "Accès à l’électricité",
    "Chomage.xml": "Chômage",
    "croissance du pib.xml": "Croissance du PIB",
    "envoi.xml": "Envois de fonds (% du PIB)",
    "Spi.xml": "SPI (score statistique)",
    "utilisateur internet.xml": "Utilisateurs d’Internet",
    # Ajoute ici d'autres fichiers : "nom_fichier.xml": "Nom de l'indicateur"
}

# === Extraire tous les fichiers et fusionner ===
dfs = []
for fichier, nom_colonne in fichiers.items():
    df = extract_indicator_from_xml(f"{fichier}", nom_colonne)
    dfs.append(df)

# === Fusionner sur l'année ===
df_final = reduce(lambda left, right: pd.merge(left, right, on="Année", how="outer"), dfs)
df_final = df_final.sort_values("Année").reset_index(drop=True)

# === Afficher ou enregistrer le résultat ===
print(df_final.head())
# df_final.to_csv("donnees_niger_fusionnees.csv", index=False)

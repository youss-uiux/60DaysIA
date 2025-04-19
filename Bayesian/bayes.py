import xml.etree.ElementTree as ET
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
    "utilisateur internet.xml": "Utilisateurs d’Internet"
}

dfs = [extract_indicator_from_xml(fichier, label) for fichier, label in fichiers.items()]
df_final = reduce(lambda left, right: pd.merge(left, right, on="Année", how="outer"), dfs)
df_final = df_final.interpolate().fillna(method='ffill').fillna(method='bfill')
df = df_final.drop(columns=["Année"])
model = bmb.Model(
    formula="`Croissance du PIB` ~ `Espérance de vie` + Chômage + `Accès à l’électricité` + `Utilisateurs d’Internet` + `Envois de fonds (% du PIB)` + `SPI (score statistique)`",
    data=df
)
results = model.fit(draws=2000, tune=1000)
az.summary(results)
az.plot_trace(results)
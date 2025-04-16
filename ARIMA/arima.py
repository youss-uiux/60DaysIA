import xml.etree.ElementTree as ET
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller

# === Extraction des données XML pour le Niger ===
def extract_gdp_growth(xml_file, country="Niger"):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    data = []
    for record in root.findall(".//record"):
        country_name = record.find("./field[@name='Country or Area']")
        year = record.find("./field[@name='Year']")
        value = record.find("./field[@name='Value']")

        if country_name is not None and year is not None and value is not None:
            if country_name.text == country and value.text:
                try:
                    data.append({"Année": int(year.text), "Croissance PIB": float(value.text)})
                except:
                    continue

    df = pd.DataFrame(data)
    df = df.sort_values("Année").reset_index(drop=True)
    df.set_index("Année", inplace=True)
    return df

# === Charger les données ===
df = extract_gdp_growth("pib.xml")

result = adfuller(df["Croissance PIB"])
print("ADF Statistic :", result[0])
print("p-value        :", result[1])

df_diff = df["Croissance PIB"].diff().dropna()

result = adfuller(df_diff)
print("ADF (diff 1) :", result[0])
print("p-value :", result[1])

# === Affichage des données ===
print(df.tail())
df.plot(title="Croissance du PIB (%) - Niger")
plt.ylabel("Croissance (%)")
plt.grid(True)
plt.show()

# === ACF & PACF ===
plot_acf(df["Croissance PIB"], lags=20)
plt.title("ACF - Croissance PIB")
plt.show()

plot_pacf(df["Croissance PIB"], lags=20)
plt.title("PACF - Croissance PIB")
plt.show()

# === Création du modèle ARIMA (exemple avec p=1, d=1, q=1) ===
model = ARIMA(df["Croissance PIB"], order=(1, 0, 1))
resultats = model.fit()

# === Affichage du résumé ===
print(resultats.summary())

# === Prédiction pour 5 années futures ===
forecast = resultats.forecast(steps=5)
forecast.index = range(df.index.max() + 1, df.index.max() + 6)

# === Affichage des prédictions ===
plt.figure(figsize=(10, 5))
plt.plot(df.index, df["Croissance PIB"], label="Historique")
plt.plot(forecast.index, forecast, label="Prévisions", linestyle='--', marker='o')
plt.title("Prévision de la Croissance du PIB (ARIMA)")
plt.xlabel("Année")
plt.ylabel("Croissance (%)")
plt.legend()
plt.grid(True)
plt.show()

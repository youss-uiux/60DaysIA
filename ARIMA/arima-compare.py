import xml.etree.ElementTree as ET
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

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

# === Modèle ARIMA(1, 0, 1)
model_d0 = ARIMA(df["Croissance PIB"], order=(1, 0, 1))
result_d0 = model_d0.fit()
residuals_d0 = result_d0.resid
mse_d0 = mean_squared_error(df["Croissance PIB"], result_d0.fittedvalues)


# === Modèle ARIMA(1, 1, 1)
model_d1 = ARIMA(df["Croissance PIB"], order=(2, 0, 2))
result_d1 = model_d1.fit()
residuals_d1 = result_d1.resid

# Couper les fittedvalues pour qu'ils aient la même taille que true_values_d1
fitted_d1 = result_d1.fittedvalues.iloc[:len(df) - 1]
true_values_d1 = df["Croissance PIB"].iloc[1:]

mse_d1 = mean_squared_error(true_values_d1, fitted_d1)


# ⚠️ Aligner la série vraie avec les prédictions (décalage pour d=1)
true_values_d1 = df["Croissance PIB"].iloc[1:]
fitted_d1 = result_d1.fittedvalues.iloc[:len(true_values_d1)]

mse_d1 = mean_squared_error(true_values_d1, fitted_d1)

# === Affichage des critères
print("=== Comparaison des modèles ===")
print(f"AIC ARIMA(1,0,1) : {result_d0.aic}")
print(f"AIC ARIMA(2,0,2) : {result_d1.aic}")
print(f"BIC ARIMA(1,0,1) : {result_d0.bic}")
print(f"BIC ARIMA(2,0,2) : {result_d1.bic}")
print(f"MSE ARIMA(1,0,1) : {mse_d0}")
print(f"MSE ARIMA(2,0,2) : {mse_d1}")

# === Graphique des résidus
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(residuals_d0, label="Résidus ARIMA(1,0,1)", color='blue')
plt.axhline(0, color='gray', linestyle='--')
plt.title("Résidus ARIMA(1,0,1)")
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(residuals_d1, label="Résidus ARIMA(2,0,2)", color='green')
plt.axhline(0, color='gray', linestyle='--')
plt.title("Résidus ARIMA(2,0,2)")
plt.grid(True)

plt.tight_layout()
plt.show()

import yfinance as yf
import pandas as pd
import numpy as np
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.estimators import BayesianEstimator, BicScore
from pgmpy.inference import VariableElimination
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# 1. Téléchargement des données
tickers = ["AAPL", "^TNX", "^VIX", "^IXIC"]
data = yf.download(tickers, start="2020-01-01", end="2023-12-31")["Close"]
data.columns = ["AAPL", "Taux_US", "VIX", "NASDAQ"]

# 2. Préparation des caractéristiques
def prepare_features(df):
    returns = df.pct_change().dropna()
    
    # Discrétisation en 3 états (-1: baisse, 0: neutre, 1: hausse)
    returns["AAPL_State"] = np.select(
        [
            returns["AAPL"] < -returns["AAPL"].std(),
            returns["AAPL"] > returns["AAPL"].std()
        ],
        [-1, 1],
        default=0
    )
    
    returns["VIX_State"] = (returns["VIX"] > returns["VIX"].median()).astype(int)
    returns["NASDAQ_State"] = (returns["NASDAQ"] > 0).astype(int)
    returns["Taux_State"] = (returns["Taux_US"] > returns["Taux_US"].median()).astype(int)
    
    return returns[["AAPL_State", "VIX_State", "NASDAQ_State", "Taux_State"]]

processed_data = prepare_features(data)

# 3. Division des données
train_data, test_data = train_test_split(processed_data, test_size=0.2, shuffle=False)

# 4. Création du modèle DiscreteBayesianNetwork
model = DiscreteBayesianNetwork([
    ("VIX_State", "AAPL_State"),
    ("NASDAQ_State", "AAPL_State"),
    ("Taux_State", "VIX_State"),
    ("Taux_State", "NASDAQ_State")
])

# 5. Apprentissage
model.fit(train_data, estimator=BayesianEstimator)

# 6. Évaluation
bic = BicScore(train_data)
print(f"Score BIC du modèle: {bic.score(model):.2f}")

# 7. Inférence
inference = VariableElimination(model)

# 8. Analyse des probabilités conditionnelles
print("\n=== Tables de probabilité ===")
for cpd in model.get_cpds():
    print(f"\n{cpd}")

# 9. Scénarios de prédiction
print("\n=== Prédictions ===")
print("Scénario 1: VIX Haut + NASDAQ Haut")
print(inference.query(["AAPL_State"], evidence={"VIX_State": 1, "NASDAQ_State": 1}))

print("\nScénario 2: Taux Bas + VIX Bas")
print(inference.query(["AAPL_State"], evidence={"Taux_State": 0, "VIX_State": 0}))

# 10. Visualisation
states = {-1: "Baisse", 0: "Neutre", 1: "Hausse"}
plt.figure(figsize=(14,6))
plt.scatter(
    processed_data.index,
    processed_data["AAPL_State"],
    c=processed_data["AAPL_State"],
    cmap="coolwarm",
    alpha=0.7
)
plt.yticks([-1, 0, 1], ["Baisse", "Neutre", "Hausse"])
plt.title("États des rendements AAPL (2020-2023)")
plt.grid()
plt.show()

# 11. Sauvegarde
model.save("apple_discrete_bn.bif")
print("\nModèle sauvegardé sous 'apple_discrete_bn.bif'")

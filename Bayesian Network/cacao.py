import yfinance as yf
import pandas as pd
import numpy as np
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.estimators import BayesianEstimator
try:
    from pgmpy.estimators import BDeuScore
except ImportError:
    BDeuScore = None  # Si BDeuScore n'est pas disponible
from pgmpy.inference import VariableElimination
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# 1. Téléchargement des données élargies
tickers = ["CC=F", "EURUSD=X", "DX-Y.NYB", "^GSPC", "^TNX", "^VIX", "CL=F"]
data = yf.download(tickers, start="2020-01-01", end="2023-12-31")["Close"]
data.columns = ["Cacao_Prix", "EURUSD", "USDX", "SP500", "Taux_US", "VIX", "Oil"]

# 2. Vérification des données téléchargées
print("Vérification des données téléchargées :")
print(data.isna().sum())  # Affiche le nombre de NaN par colonne

# Gérer les valeurs manquantes avec forward fill
data = data.fillna(method="ffill").fillna(method="bfill")  # Remplissage avant et arrière

# 3. Préparation des caractéristiques
def prepare_features(df):
    # Calcul des rendements avec fill_method=None pour éviter l'avertissement
    returns = df.pct_change(fill_method=None).dropna()
    
    # Vérifier si le DataFrame est vide après dropna
    if returns.empty:
        raise ValueError("Le DataFrame des rendements est vide après suppression des NaN.")
    
    # Discrétisation en 3 états (-1: baisse, 0: neutre, 1: hausse) pour Cacao
    returns["Cacao_State"] = np.select(
        [
            returns["Cacao_Prix"] < -returns["Cacao_Prix"].std(),
            returns["Cacao_Prix"] > returns["Cacao_Prix"].std()
        ],
        [-1, 1],
        default=0
    )
    
    # Discrétisation binaire pour les autres variables (bas/haut)
    returns["EURUSD_State"] = (returns["EURUSD"] > returns["EURUSD"].median()).astype(int)
    returns["USDX_State"] = (returns["USDX"] > returns["USDX"].median()).astype(int)
    returns["SP500_State"] = (returns["SP500"] > 0).astype(int)
    returns["Taux_State"] = (returns["Taux_US"] > returns["Taux_US"].median()).astype(int)
    returns["VIX_State"] = (returns["VIX"] > returns["VIX"].median()).astype(int)
    returns["Oil_State"] = (returns["Oil"] > returns["Oil"].median()).astype(int)
    
    return returns[["Cacao_State", "EURUSD_State", "USDX_State", "SP500_State", "Taux_State", "VIX_State", "Oil_State"]]

try:
    processed_data = prepare_features(data)
    print(f"Données traitées : {processed_data.shape[0]} lignes")
except ValueError as e:
    print(f"Erreur lors du prétraitement : {e}")
    exit(1)

# 4. Division des données
try:
    train_data, test_data = train_test_split(processed_data, test_size=0.2, shuffle=False)
    print(f"Données d'entraînement : {train_data.shape[0]} lignes")
    print(f"Données de test : {test_data.shape[0]} lignes")
except ValueError as e:
    print(f"Erreur lors de la division des données : {e}")
    exit(1)

# 5. Création du modèle DiscreteBayesianNetwork
model = DiscreteBayesianNetwork([
    ("Taux_State", "VIX_State"),
    ("Taux_State", "SP500_State"),
    ("VIX_State", "Cacao_State"),
    ("SP500_State", "Cacao_State"),
    ("EURUSD_State", "Cacao_State"),
    ("USDX_State", "Cacao_State"),
    ("Oil_State", "Cacao_State")
])

# 6. Apprentissage
model.fit(train_data, estimator=BayesianEstimator, prior_type="BDeu")

# 7. Évaluation (si BDeuScore est disponible)
if BDeuScore is not None:
    bdeu = BDeuScore(train_data)
    print(f"Score BDeu du modèle: {bdeu.score(model):.2f}")
else:
    print("Évaluation du score non disponible (BDeuScore non trouvé).")

# 8. Inférence
inference = VariableElimination(model)

# 9. Analyse des probabilités conditionnelles
print("\n=== Tables de probabilité ===")
for cpd in model.get_cpds():
    print(f"\n{cpd}")

# 10. Scénarios de prédiction
print("\n=== Prédictions ===")
print("Scénario 1: VIX Haut + SP500 Haut + EURUSD Haut")
print(inference.query(["Cacao_State"], evidence={"VIX_State": 1, "SP500_State": 1, "EURUSD_State": 1}))

print("\nScénario 2: Taux Bas + Oil Bas + USDX Bas")
print(inference.query(["Cacao_State"], evidence={"Taux_State": 0, "Oil_State": 0, "USDX_State": 0}))

# 11. Visualisation
states = {-1: "Baisse", 0: "Neutre", 1: "Hausse"}
plt.figure(figsize=(14, 6))
plt.scatter(
    processed_data.index,
    processed_data["Cacao_State"],
    c=processed_data["Cacao_State"],
    cmap="coolwarm",
    alpha=0.7
)
plt.yticks([-1, 0, 1], ["Baisse", "Neutre", "Hausse"])
plt.title("États des rendements du prix du cacao (2020-2023)")
plt.grid()
plt.savefig("cacao_states_extended_v2_plot.png")
print("Visualisation sauvegardée sous 'cacao_states_extended_v2_plot.png'")

# 12. Sauvegarde
model.save("cacao_extended_v2_bn.bif")
print("\nModèle sauvegardé sous 'cacao_extended_v2_bn.bif'")
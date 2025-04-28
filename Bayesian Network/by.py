from typing import Union  # Pour la compatibilité Python < 3.10
import yfinance as yf
import pandas as pd
import numpy as np
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination
import matplotlib.pyplot as plt

# 1. Téléchargement des données
try:
    tickers = ["AAPL", "^TNX", "DX-Y.NYB"]
    data = yf.download(tickers, start="2020-01-01", end="2023-12-31")["Close"]
    data.columns = ["AAPL", "Taux_US", "Dollar_Index"]
    print("Données téléchargées avec succès!")
except Exception as e:
    print(f"Erreur lors du téléchargement des données: {e}")
    exit()

# 2. Préparation des données
try:
    returns = data.pct_change().dropna()
    returns["AAPL_Up"] = (returns["AAPL"] > 0).astype(int)
    returns["Taux_Up"] = (returns["Taux_US"] > 0).astype(int)
    returns["Dollar_Up"] = (returns["Dollar_Index"] > 0).astype(int)
except Exception as e:
    print(f"Erreur lors du prétraitement: {e}")
    exit()

# 3. Construction du modèle
try:
    model = DiscreteBayesianNetwork([
        ("Taux_Up", "Dollar_Up"),
        ("Dollar_Up", "AAPL_Up"), 
        ("Taux_Up", "AAPL_Up")
    ])
    print("Réseau bayésien créé avec succès!")
except Exception as e:
    print(f"Erreur lors de la création du modèle: {e}")
    exit()

# 4. Apprentissage
try:
    model.fit(returns[["Taux_Up", "Dollar_Up", "AAPL_Up"]], 
             estimator=MaximumLikelihoodEstimator)
    print("Modèle entraîné avec succès!")
except Exception as e:
    print(f"Erreur lors de l'apprentissage: {e}")
    exit()

# 5. Analyse
print("\n=== Structure du réseau ===")
print(model.edges())

print("\n=== Tables de probabilité ===")
for cpd in model.get_cpds():
    print(f"\n{cpd}")

# 6. Inférence
try:
    inference = VariableElimination(model)
    
    # Scénario 1
    proba1 = inference.query(variables=["AAPL_Up"], 
                           evidence={"Taux_Up": 1, "Dollar_Up": 1})
    print(f"\nProbabilité AAPL ↗ (taux ↗ + dollar ↗): {proba1.values[1]:.1%}")
    
    # Scénario 2
    proba2 = inference.query(variables=["AAPL_Up"], 
                           evidence={"Taux_Up": 1, "Dollar_Up": 0})
    print(f"Probabilité AAPL ↗ (taux ↗ + dollar ↘): {proba2.values[1]:.1%}")
    
except Exception as e:
    print(f"Erreur lors de l'inférence: {e}")

# 7. Visualisation
try:
    plt.figure(figsize=(12, 5))
    plt.plot(data.index, data["AAPL"], label='Prix AAPL', color='blue')
    plt.title('Prix historique d\'Apple (2020-2023)')
    plt.xlabel('Date')
    plt.ylabel('Prix ($)')
    plt.legend()
    plt.grid(True)
    plt.show()
except Exception as e:
    print(f"Erreur lors de la visualisation: {e}")

# 8. Sauvegarde
try:
    model.save("modele_AAPL.bif")
    print("\nModèle sauvegardé sous 'modele_AAPL.bif'")
except Exception as e:
    print(f"Erreur lors de la sauvegarde: {e}")
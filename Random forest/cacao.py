import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Téléchargement des données
tickers = ["CC=F", "EURUSD=X", "DX-Y.NYB", "^GSPC", "^TNX", "^VIX", "CL=F"]
data = yf.download(tickers, start="2020-01-01", end="2023-12-31")["Close"]
data.columns = ["Cacao_Prix", "EURUSD", "USDX", "SP500", "Taux_US", "VIX", "Oil"]

# 2. Gestion des valeurs manquantes
print("Vérification des NaN :")
print(data.isna().sum())
data = data.fillna(method="ffill").fillna(method="bfill")

# 3. Préparation des caractéristiques
def prepare_features(df, lags=5):
    # Calculer les rendements
    returns = df.pct_change(fill_method=None).dropna()
    
    # Créer des caractéristiques décalées et moyenne mobile
    features = pd.DataFrame()
    for col in df.columns:
        for lag in range(1, lags + 1):
            features[f"{col}_lag{lag}"] = returns[col].shift(lag)
        features[f"{col}_ma5"] = returns[col].rolling(window=5).mean()
    
    # Discrétiser la cible (Cacao_State : -1, 0, 1)
    features["Cacao_State"] = np.select(
        [
            returns["Cacao_Prix"] < -returns["Cacao_Prix"].std(),
            returns["Cacao_Prix"] > returns["Cacao_Prix"].std()
        ],
        [-1, 1],
        default=0
    )
    
    # Supprimer les NaN
    features = features.dropna()
    
    return features

# Préparer les données
processed_data = prepare_features(data)
if processed_data.empty:
    raise ValueError("Les données traitées sont vides.")

# 4. Séparation des features et cible
X = processed_data.drop("Cacao_State", axis=1)
y = processed_data["Cacao_State"]

# 5. Division entraînement/test (séries temporelles)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
print(f"Données d'entraînement : {X_train.shape[0]} lignes")
print(f"Données de test : {X_test.shape[0]} lignes")

# 6. Entraînement du modèle avec class_weight
rf_model = RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42)
rf_model.fit(X_train, y_train)

# 7. Validation croisée
scores = cross_val_score(rf_model, X_train, y_train, cv=5)
print(f"Validation croisée : {scores.mean():.2f} ± {scores.std():.2f}")

# 8. Prédictions et évaluation
y_pred = rf_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\nPrécision sur l'ensemble de test : {accuracy:.2f}")
print("\nRapport de classification :")
print(classification_report(y_test, y_pred, target_names=["Baisse", "Neutre", "Hausse"]))

# 9. Matrice de confusion
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Baisse", "Neutre", "Hausse"], yticklabels=["Baisse", "Neutre", "Hausse"])
plt.title("Matrice de confusion - Forêt Aléatoire")
plt.xlabel("Prédit")
plt.ylabel("Réel")
plt.savefig("rf_confusion_matrix_practice.png")
print("Matrice de confusion sauvegardée sous 'rf_confusion_matrix_practice.png'")

# 10. Importance des caractéristiques
importances = rf_model.feature_importances_
feature_names = X.columns
feature_importance_df = pd.DataFrame({"Feature": feature_names, "Importance": importances}).sort_values("Importance", ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x="Importance", y="Feature", data=feature_importance_df.head(10))
plt.title("Top 10 des caractéristiques les plus importantes")
plt.savefig("rf_feature_importance_practice.png")
print("Importance des caractéristiques sauvegardée sous 'rf_feature_importance_practice.png'")

# 11. Visualisation des prédictions
plt.figure(figsize=(14, 6))
plt.plot(y_test.index, y_test, label="Réel", marker="o", alpha=0.6)
plt.plot(y_test.index, y_pred, label="Prédit", marker="x", alpha=0.6)
plt.title("Prédictions vs Réel (Cacao_State)")
plt.yticks([-1, 0, 1], ["Baisse", "Neutre", "Hausse"])
plt.legend()
plt.grid()
plt.savefig("rf_predictions_practice.png")
print("Prédictions sauvegardées sous 'rf_predictions_practice.png'")
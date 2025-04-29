import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Charger les données (remplace par ton chemin)
data = pd.read_csv("sales.csv")  # Assure-toi d'avoir le fichier

# 2. Prétraitement
data["Date"] = pd.to_datetime(data["Date"])
data = data[~data["TransactionNo"].str.startswith("C")]
data = data[data["Quantity"] >= 0]

# Agréger par client
customer_data = data.groupby("CustomerNo").agg({
    "TransactionNo": "count",
    "Price": ["mean", "var"],
    "Quantity": "sum",
    "Country": "first",
    "Date": ["min", "max"]
}).reset_index()

customer_data.columns = ["CustomerNo", "TotalTransactions", "AvgPrice", "PriceVariance", "TotalQuantity", "Country", "FirstPurchase", "LastPurchase"]

# Calculer des features
customer_data["TenureDays"] = (customer_data["LastPurchase"] - customer_data["FirstPurchase"]).dt.days
customer_data["TransactionsPerMonth"] = customer_data["TotalTransactions"] / (customer_data["TenureDays"] / 30).clip(lower=1)
customer_data["PriceVariance"] = customer_data["PriceVariance"].fillna(0)

# Encoder Country
le = LabelEncoder()
customer_data["Country_Encoded"] = le.fit_transform(customer_data["Country"])

# Définir la fidélité (moins déterministe)
customer_data["Loyalty"] = (customer_data["TransactionsPerMonth"] > 1).astype(int)  # Fidèle si plus de 1 transaction par mois

# 3. Sélection des features (sans TotalTransactions ni AvgSpend)
features = [ "PriceVariance", "TotalQuantity", "TenureDays", "TransactionsPerMonth", "Country_Encoded"]
X = customer_data[features]
y = customer_data["Loyalty"]

# 4. Vérification des données
print("Répartition des classes :")
print(y.value_counts())

# 5. Division entraînement/test (split temporel)
split_idx = int(len(X) * 0.8)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
print(f"\nDonnées d'entraînement : {X_train.shape[0]} lignes")
print(f"Données de test : {X_test.shape[0]} lignes")

# 6. Entraînement du modèle
rf_model = RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42)
rf_model.fit(X_train, y_train)

# 7. Prédictions et évaluation
y_pred = rf_model.predict(X_test)
print("\nRapport de classification :")
print(classification_report(y_test, y_pred, target_names=["Occasionnel", "Fidèle"]))

# 8. Matrice de confusion
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Occasionnel", "Fidèle"], yticklabels=["Occasionnel", "Fidèle"])
plt.title("Matrice de confusion - Prédiction de la fidélité client")
plt.xlabel("Prédit")
plt.ylabel("Réel")
plt.savefig("loyalty_confusion_matrix_corrected.png")
print("Matrice de confusion sauvegardée sous 'loyalty_confusion_matrix_corrected.png'")

# 9. Importance des caractéristiques
importances = rf_model.feature_importances_
feature_names = X.columns
feature_importance_df = pd.DataFrame({"Feature": feature_names, "Importance": importances}).sort_values("Importance", ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x="Importance", y="Feature", data=feature_importance_df)
plt.title("Importance des caractéristiques")
plt.savefig("loyalty_feature_importance_corrected.png")
print("Importance des caractéristiques sauvegardée sous 'loyalty_feature_importance_corrected.png'")

# 10. Courbe ROC
y_pred_proba = rf_model.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color="#1E3A8A", label=f"ROC curve (AUC = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], color="gray", linestyle="--")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("Taux de faux positifs")
plt.ylabel("Taux de vrais positifs")
plt.title("Courbe ROC - Prédiction de la fidélité client")
plt.legend(loc="lower right")
plt.grid()
plt.savefig("loyalty_roc_curve_corrected.png")
print("Courbe ROC sauvegardée sous 'loyalty_roc_curve_corrected.png'")
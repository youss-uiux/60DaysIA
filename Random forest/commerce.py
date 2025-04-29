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
# Identifier les annulations (uniquement avec TransactionNo)
data["Cancellation"] = data["TransactionNo"].str.startswith("C").astype(int)

# Convertir Date en datetime et extraire des features
data["Date"] = pd.to_datetime(data["Date"])
data["Month"] = data["Date"].dt.month
data["DayOfWeek"] = data["Date"].dt.dayofweek
data["IsWeekend"] = data["DayOfWeek"].isin([5, 6]).astype(int)

# Encoder Country
le = LabelEncoder()
data["Country_Encoded"] = le.fit_transform(data["Country"])

# Calculer des features client
customer_freq = data.groupby("CustomerNo")["TransactionNo"].count().reset_index()
customer_freq.columns = ["CustomerNo", "PurchaseFrequency"]
data = data.merge(customer_freq, on="CustomerNo", how="left")

# Gérer les NaN
data = data.dropna(subset=["CustomerNo", "Price"])

# 3. Sélection des features (sans Quantity)
features = ["Price", "Month", "DayOfWeek", "IsWeekend", "Country_Encoded", "PurchaseFrequency"]
X = data[features]
y = data["Cancellation"]

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
print(classification_report(y_test, y_pred, target_names=["Non annulée", "Annulée"]))

# 8. Matrice de confusion
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Non annulée", "Annulée"], yticklabels=["Non annulée", "Annulée"])
plt.title("Matrice de confusion - Prédiction des annulations")
plt.xlabel("Prédit")
plt.ylabel("Réel")
plt.savefig("cancellation_confusion_matrix_corrected.png")
print("Matrice de confusion sauvegardée sous 'cancellation_confusion_matrix_corrected.png'")

# 9. Importance des caractéristiques
importances = rf_model.feature_importances_
feature_names = X.columns
feature_importance_df = pd.DataFrame({"Feature": feature_names, "Importance": importances}).sort_values("Importance", ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x="Importance", y="Feature", data=feature_importance_df)
plt.title("Importance des caractéristiques")
plt.savefig("cancellation_feature_importance_corrected.png")
print("Importance des caractéristiques sauvegardée sous 'cancellation_feature_importance_corrected.png'")

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
plt.title("Courbe ROC - Prédiction des annulations")
plt.legend(loc="lower right")
plt.grid()
plt.savefig("cancellation_roc_curve_corrected.png")
print("Courbe ROC sauvegardée sous 'cancellation_roc_curve_corrected.png'")
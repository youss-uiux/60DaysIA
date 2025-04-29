import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Générer un dataset synthétique
np.random.seed(42)
n_samples = 10000
data = pd.DataFrame({
    "montant": np.random.uniform(1, 1000, n_samples),  # Montant de la transaction
    "heure": np.random.uniform(0, 24, n_samples),  # Heure de la journée
    "distance_pays": np.random.uniform(0, 5000, n_samples),  # Distance du pays habituel (km)
    "achats_passes": np.random.randint(0, 10, n_samples),  # Nb d'achats récents
    "frequence_jour": np.random.uniform(0, 5, n_samples),  # Transactions par jour
    "fraude": np.random.choice([0, 1], n_samples, p=[0.99, 0.01])  # 1% de fraudes
})

# Simuler des fraudes réalistes (montants élevés, distances inhabituelles)
fraud_idx = data[data["fraude"] == 1].index
data.loc[fraud_idx, "montant"] = np.random.uniform(500, 2000, len(fraud_idx))
data.loc[fraud_idx, "distance_pays"] = np.random.uniform(2000, 6000, len(fraud_idx))

# 2. Vérification des données
print("Aperçu des données :")
print(data.head())
print("\nRépartition des classes :")
print(data["fraude"].value_counts())

# 3. Séparation des features et cible
X = data.drop("fraude", axis=1)
y = data["fraude"]

# 4. Division entraînement/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
print(f"\nDonnées d'entraînement : {X_train.shape[0]} lignes")
print(f"Données de test : {X_test.shape[0]} lignes")

# 5. Entraînement du modèle
rf_model = RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42)
rf_model.fit(X_train, y_train)

# 6. Prédictions et évaluation
y_pred = rf_model.predict(X_test)
print("\nRapport de classification :")
print(classification_report(y_test, y_pred, target_names=["Légitime", "Fraude"]))

# 7. Matrice de confusion
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Légitime", "Fraude"], yticklabels=["Légitime", "Fraude"])
plt.title("Matrice de confusion - Détection de fraudes")
plt.xlabel("Prédit")
plt.ylabel("Réel")
plt.savefig("fraud_confusion_matrix.png")
print("Matrice de confusion sauvegardée sous 'fraud_confusion_matrix.png'")

# 8. Importance des caractéristiques
importances = rf_model.feature_importances_
feature_names = X.columns
feature_importance_df = pd.DataFrame({"Feature": feature_names, "Importance": importances}).sort_values("Importance", ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x="Importance", y="Feature", data=feature_importance_df)
plt.title("Importance des caractéristiques")
plt.savefig("fraud_feature_importance.png")
print("Importance des caractéristiques sauvegardée sous 'fraud_feature_importance.png'")

# 9. Courbe ROC
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
plt.title("Courbe ROC - Détection de fraudes")
plt.legend(loc="lower right")
plt.grid()
plt.savefig("fraud_roc_curve.png")
print("Courbe ROC sauvegardée sous 'fraud_roc_curve.png'")
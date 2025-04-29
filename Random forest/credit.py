import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Charger les données (remplace par ton chemin)
data = pd.read_csv("creditcard.csv")  # Assure-toi d'avoir le fichier

# 2. Prétraitement
# Vérifier les valeurs manquantes
print("Valeurs manquantes :")
print(data.isnull().sum())

# Standardiser Time et Amount
scaler = StandardScaler()
data["Time"] = scaler.fit_transform(data[["Time"]])
data["Amount"] = scaler.fit_transform(data[["Amount"]])

# 3. Sélection des features et cible
features = [col for col in data.columns if col != "Class"]  # Toutes les colonnes sauf Class
X = data[features]
y = data["Class"]

# 4. Vérification des données
print("\nRépartition des classes :")
print(y.value_counts(normalize=True))

# 5. Division entraînement/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
print(f"\nDonnées d'entraînement : {X_train.shape[0]} lignes")
print(f"Données de test : {X_test.shape[0]} lignes")

# 6. Entraînement du modèle
rf_model = RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42)
rf_model.fit(X_train, y_train)

# 7. Prédictions et évaluation
y_pred = rf_model.predict(X_test)
print("\nRapport de classification :")
print(classification_report(y_test, y_pred, target_names=["Légitime", "Fraude"]))

# 8. Matrice de confusion
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Légitime", "Fraude"], yticklabels=["Légitime", "Fraude"])
plt.title("Matrice de confusion - Détection de fraudes")
plt.xlabel("Prédit")
plt.ylabel("Réel")
plt.savefig("creditcard_confusion_matrix.png")
print("Matrice de confusion sauvegardée sous 'creditcard_confusion_matrix.png'")

# 9. Importance des caractéristiques
importances = rf_model.feature_importances_
feature_names = X.columns
feature_importance_df = pd.DataFrame({"Feature": feature_names, "Importance": importances}).sort_values("Importance", ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x="Importance", y="Feature", data=feature_importance_df.head(10))  # Top 10 features
plt.title("Importance des caractéristiques (Top 10)")
plt.savefig("creditcard_feature_importance.png")
print("Importance des caractéristiques sauvegardée sous 'creditcard_feature_importance.png'")

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
plt.title("Courbe ROC - Détection de fraudes")
plt.legend(loc="lower right")
plt.grid()
plt.savefig("creditcard_roc_curve.png")
print("Courbe ROC sauvegardée sous 'creditcard_roc_curve.png'")

# 11. Ajuster le seuil pour maximiser le rappel
threshold = 0.3  # Abaisser le seuil pour augmenter le rappel
y_pred_adjusted = (y_pred_proba >= threshold).astype(int)
print("\nRapport de classification avec seuil ajusté (0.3) :")
print(classification_report(y_test, y_pred_adjusted, target_names=["Légitime", "Fraude"]))
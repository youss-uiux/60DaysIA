import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import StackingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import recall_score, precision_score, roc_auc_score, accuracy_score, confusion_matrix, roc_curve
from sklearn.preprocessing import StandardScaler

# Charger le dataset
df = pd.read_csv("creditcard.csv")

# Séparer features et cible
X = df.drop(columns=['Class'])
y = df['Class']

# Diviser l'ensemble complet en train/test (avant équilibrage)
X_train_full, X_test_full, y_train_full, y_test_full = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Vérifier la répartition des fraudes dans l'ensemble de test
print(f"Fraudes dans l'ensemble de test complet : {sum(y_test_full)} ({sum(y_test_full)/len(y_test_full)*100:.2f}%)")

# Sous-échantillonner l'ensemble d'entraînement pour équilibrer
df_train = pd.concat([X_train_full, y_train_full], axis=1)
fraud_train = df_train[df_train['Class'] == 1]
non_fraud_train = df_train[df_train['Class'] == 0].sample(len(fraud_train) * 2, random_state=42)
df_train_balanced = pd.concat([fraud_train, non_fraud_train])
X_train_balanced = df_train_balanced.drop(columns=['Class'])
y_train_balanced = df_train_balanced['Class']

# Standardiser les features
scaler = StandardScaler()
X_train_balanced = scaler.fit_transform(X_train_balanced)
X_test_full = scaler.transform(X_test_full)

# Diviser l'ensemble équilibré pour comparaison (comme dans ton test initial)
X_train, X_test_subset, y_train, y_test_subset = train_test_split(X_train_balanced, y_train_balanced, test_size=0.3, random_state=42)

# Définir les modèles de base
base_models = [
    ('lr', LogisticRegression(max_iter=200, random_state=42)),
    ('gnb', GaussianNB()),
    ('xgb', XGBClassifier(learning_rate=0.05, n_estimators=200, max_depth=4, random_state=42))
]

# Définir le modèle d'ensemble (stacking)
stacking_model = StackingClassifier(estimators=base_models, final_estimator=LogisticRegression(), cv=5)

# Entraîner le modèle
stacking_model.fit(X_train, y_train)

# --- Résultats sur le sous-ensemble équilibré ---
y_pred_subset = stacking_model.predict(X_test_subset)
y_pred_proba_subset = stacking_model.predict_proba(X_test_subset)[:, 1]

recall_subset = recall_score(y_test_subset, y_pred_subset)
precision_subset = precision_score(y_test_subset, y_pred_subset)
auc_subset = roc_auc_score(y_test_subset, y_pred_proba_subset)
accuracy_subset = accuracy_score(y_test_subset, y_pred_subset)
cm_subset = confusion_matrix(y_test_subset, y_pred_subset)

print("Métriques sur le sous-ensemble équilibré (test) :")
print(f"Rappel = {recall_subset:.2f}, Précision = {precision_subset:.2f}, AUC = {auc_subset:.2f}, Exactitude = {accuracy_subset:.2f}")
print("Matrice de confusion (sous-ensemble équilibré) :")
print(cm_subset)

# Vérifier le surapprentissage
y_train_pred = stacking_model.predict(X_train)
y_train_pred_proba = stacking_model.predict_proba(X_train)[:, 1]
recall_train = recall_score(y_train, y_train_pred)
precision_train = precision_score(y_train, y_train_pred)
auc_train = roc_auc_score(y_train, y_train_pred_proba)
accuracy_train = accuracy_score(y_train, y_train_pred)

print("\nMétriques sur l'ensemble d'entraînement (vérification surapprentissage) :")
print(f"Rappel = {recall_train:.2f}, Précision = {precision_train:.2f}, AUC = {auc_train:.2f}, Exactitude = {accuracy_train:.2f}")

# Graphique : Matrice de confusion (sous-ensemble équilibré)
plt.figure(figsize=(6, 5))
sns.heatmap(cm_subset, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Non-Fraude', 'Fraude'], 
            yticklabels=['Non-Fraude', 'Fraude'])
plt.title("Matrice de Confusion - Sous-ensemble Équilibré")
plt.xlabel("Prédictions")
plt.ylabel("Réel")
plt.savefig("confusion_matrix_subset.png")
plt.show()

# Graphique : Courbe ROC (sous-ensemble équilibré)
fpr, tpr, _ = roc_curve(y_test_subset, y_pred_proba_subset)
plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, label=f'Stacking (AUC = {auc_subset:.2f})', color='green')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('Taux de faux positifs')
plt.ylabel('Taux de vrais positifs')
plt.title('Courbe ROC : Sous-ensemble Équilibré')
plt.legend()
plt.savefig("roc_curve_subset.png")
plt.show()

# --- Résultats sur l'ensemble complet (non équilibré) ---
y_pred_full = stacking_model.predict(X_test_full)
y_pred_proba_full = stacking_model.predict_proba(X_test_full)[:, 1]

recall_full = recall_score(y_test_full, y_pred_full)
precision_full = precision_score(y_test_full, y_pred_full)
auc_full = roc_auc_score(y_test_full, y_pred_proba_full)
accuracy_full = accuracy_score(y_test_full, y_pred_full)
cm_full = confusion_matrix(y_test_full, y_pred_full)

print("\nMétriques sur l'ensemble complet (non équilibré) :")
print(f"Rappel = {recall_full:.2f}, Précision = {precision_full:.2f}, AUC = {auc_full:.2f}, Exactitude = {accuracy_full:.2f}")
print("Matrice de confusion (ensemble complet) :")
print(cm_full)

# Graphique : Matrice de confusion (ensemble complet)
plt.figure(figsize=(6, 5))
sns.heatmap(cm_full, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Non-Fraude', 'Fraude'], 
            yticklabels=['Non-Fraude', 'Fraude'])
plt.title("Matrice de Confusion - Ensemble Complet")
plt.xlabel("Prédictions")
plt.ylabel("Réel")
plt.savefig("confusion_matrix_full.png")
plt.show()

# Graphique : Courbe ROC (ensemble complet)
fpr, tpr, _ = roc_curve(y_test_full, y_pred_proba_full)
plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, label=f'Stacking (AUC = {auc_full:.2f})', color='green')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('Taux de faux positifs')
plt.ylabel('Taux de vrais positifs')
plt.title('Courbe ROC : Ensemble Complet')
plt.legend()
plt.savefig("roc_curve_full.png")
plt.show()
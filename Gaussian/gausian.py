import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import recall_score, roc_auc_score, accuracy_score, confusion_matrix, roc_curve
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

# 1. Charger le dataset
df = pd.read_csv("diabetes.csv")

# 2. Nettoyer les données (remplacer 0 par NaN, puis imputer par la médiane)
medical_features = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
df[medical_features] = df[medical_features].replace(0, np.nan)
df.fillna(df.median(), inplace=True)

# 3. Séparer features et cible
X = df.drop('Outcome', axis=1)
y = df['Outcome']

# 4. Normaliser les features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 5. Diviser en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# 6. Entraîner GNB
gnb = GaussianNB()
gnb.fit(X_train, y_train)

# 7. Faire des prédictions avec GNB
y_pred_gnb = gnb.predict(X_test)
y_pred_proba_gnb = gnb.predict_proba(X_test)[:, 1]

# 8. Calculer les métriques pour GNB
recall_gnb = recall_score(y_test, y_pred_gnb)
auc_gnb = roc_auc_score(y_test, y_pred_proba_gnb)
accuracy_gnb = accuracy_score(y_test, y_pred_gnb)
cm_gnb = confusion_matrix(y_test, y_pred_gnb)
print(f"Métriques GNB : Rappel = {recall_gnb:.2f}, AUC = {auc_gnb:.2f}, Précision = {accuracy_gnb:.2f}")

# 9. Entraîner XGBoost
xgb = XGBClassifier(learning_rate=0.05, n_estimators=200, max_depth=4, subsample=0.9,
                    colsample_bytree=0.7, gamma=0.1, eval_metric='logloss', random_state=42)
xgb.fit(X_train, y_train)

# 10. Faire des prédictions avec XGBoost
y_pred_xgb = xgb.predict(X_test)
y_pred_proba_xgb = xgb.predict_proba(X_test)[:, 1]

# 11. Calculer les métriques pour XGBoost
recall_xgb = recall_score(y_test, y_pred_xgb)
auc_xgb = roc_auc_score(y_test, y_pred_proba_xgb)
accuracy_xgb = accuracy_score(y_test, y_pred_xgb)
cm_xgb = confusion_matrix(y_test, y_pred_xgb)
print(f"Métriques XGBoost : Rappel = {recall_xgb:.2f}, AUC = {auc_xgb:.2f}, Précision = {accuracy_xgb:.2f}")

# 12. Graphique 1 : Comparaison des métriques (barres)
plt.figure(figsize=(10, 6))
models = ['GNB', 'XGBoost']
metrics = ['Recall', 'AUC', 'Accuracy']
x = np.arange(len(models))
width = 0.25
plt.bar(x - width, [recall_gnb, recall_xgb], width, label='Recall', color='skyblue')
plt.bar(x, [auc_gnb, auc_xgb], width, label='AUC', color='lightgreen')
plt.bar(x + width, [accuracy_gnb, accuracy_xgb], width, label='Accuracy', color='salmon')
plt.xlabel('Modèles')
plt.ylabel('Scores')
plt.title('Comparaison des performances : GNB vs XGBoost')
plt.xticks(x, models)
plt.legend()
plt.savefig("metrics_comparison.png")
plt.show()

# 13. Graphique 2 : Comparaison des matrices de confusion (côte à côte)
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.heatmap(cm_gnb, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Non diabétique', 'Diabétique'], 
            yticklabels=['Non diabétique', 'Diabétique'])
plt.title("Matrice de confusion - GNB")
plt.xlabel("Prédictions")
plt.ylabel("Valeurs réelles")

plt.subplot(1, 2, 2)
sns.heatmap(cm_xgb, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Non diabétique', 'Diabétique'], 
            yticklabels=['Non diabétique', 'Diabétique'])
plt.title("Matrice de confusion - XGBoost")
plt.xlabel("Prédictions")
plt.ylabel("Valeurs réelles")
plt.tight_layout()
plt.savefig("confusion_matrices_comparison.png")
plt.show()

# 14. Graphique 3 : Courbe ROC superposée
plt.figure(figsize=(8, 6))
fpr_gnb, tpr_gnb, _ = roc_curve(y_test, y_pred_proba_gnb)
plt.plot(fpr_gnb, tpr_gnb, label=f'GNB (AUC = {auc_gnb:.2f})', color='blue')
fpr_xgb, tpr_xgb, _ = roc_curve(y_test, y_pred_proba_xgb)
plt.plot(fpr_xgb, tpr_xgb, label=f'XGBoost (AUC = {auc_xgb:.2f})', color='orange')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('Taux de faux positifs')
plt.ylabel('Taux de vrais positifs')
plt.title('Courbe ROC : GNB vs XGBoost')
plt.legend()
plt.savefig("roc_curve_comparison.png")
plt.show()
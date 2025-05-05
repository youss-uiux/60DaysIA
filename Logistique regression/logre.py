import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import recall_score, roc_auc_score, accuracy_score, confusion_matrix, roc_curve
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
import nltk
nltk.download('stopwords')

# Charger les datasets
train = pd.read_csv("twitter_training.csv", header=None)
val = pd.read_csv("twitter_validation.csv", header=None)

# Renommer les colonnes
train.columns = ['id', 'information', 'type', 'text']
val.columns = ['id', 'information', 'type', 'text']

# Combiner train et validation pour plus de données (optionnel)
df = pd.concat([train, val], ignore_index=True)

# Nettoyer et préparer les données
# Convertir 'type' en binaire : 1 pour "Positive" (drôle), 0 pour autres
df['sentiment'] = df['type'].apply(lambda x: 1 if x == "Positive" else 0)

# Nettoyer la colonne 'text' pour supprimer les NaN ou non-strings
df = df.dropna(subset=['text'])
df = df[df['text'].apply(lambda x: isinstance(x, str))]

# Préparer les données pour TF-IDF
tfidf = TfidfVectorizer(max_features=500, stop_words=nltk.corpus.stopwords.words('english'))
X = tfidf.fit_transform(df['text'])
y = df['sentiment']

# Diviser en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Entraîner Logistic Regression
lr = LogisticRegression(max_iter=200, random_state=42)
lr.fit(X_train, y_train)

# Prédictions avec Logistic Regression
y_pred_lr = lr.predict(X_test)
y_pred_proba_lr = lr.predict_proba(X_test)[:, 1]

# Calculer les métriques pour Logistic Regression
recall_lr = recall_score(y_test, y_pred_lr)
auc_lr = roc_auc_score(y_test, y_pred_proba_lr)
accuracy_lr = accuracy_score(y_test, y_pred_lr)
cm_lr = confusion_matrix(y_test, y_pred_lr)
print(f"Métriques Régression Logistique : Rappel = {recall_lr:.2f}, AUC = {auc_lr:.2f}, Précision = {accuracy_lr:.2f}")

# Entraîner GNB pour comparaison
gnb = GaussianNB()
gnb.fit(X_train.toarray(), y_train)
y_pred_gnb = gnb.predict(X_test.toarray())
y_pred_proba_gnb = gnb.predict_proba(X_test.toarray())[:, 1]
recall_gnb = recall_score(y_test, y_pred_gnb)
auc_gnb = roc_auc_score(y_test, y_pred_proba_gnb)
accuracy_gnb = accuracy_score(y_test, y_pred_gnb)
cm_gnb = confusion_matrix(y_test, y_pred_gnb)
print(f"Métriques GNB : Rappel = {recall_gnb:.2f}, AUC = {auc_gnb:.2f}, Précision = {accuracy_gnb:.2f}")

# Entraîner XGBoost pour comparaison
xgb = XGBClassifier(learning_rate=0.05, n_estimators=200, max_depth=4, random_state=42)
xgb.fit(X_train, y_train)
y_pred_xgb = xgb.predict(X_test)
y_pred_proba_xgb = xgb.predict_proba(X_test)[:, 1]
recall_xgb = recall_score(y_test, y_pred_xgb)
auc_xgb = roc_auc_score(y_test, y_pred_proba_xgb)
accuracy_xgb = accuracy_score(y_test, y_pred_xgb)
cm_xgb = confusion_matrix(y_test, y_pred_xgb)
print(f"Métriques XGBoost : Rappel = {recall_xgb:.2f}, AUC = {auc_xgb:.2f}, Précision = {accuracy_xgb:.2f}")

# Graphique 1 : Comparaison des métriques (barres)
plt.figure(figsize=(10, 6))
models = ['GNB', 'XGBoost', 'LogReg']
x = np.arange(len(models))
width = 0.25
plt.bar(x - width, [recall_gnb, recall_xgb, recall_lr], width, label='Recall', color='skyblue')
plt.bar(x, [auc_gnb, auc_xgb, auc_lr], width, label='AUC', color='lightgreen')
plt.bar(x + width, [accuracy_gnb, accuracy_xgb, accuracy_lr], width, label='Accuracy', color='salmon')
plt.xlabel('Modèles')
plt.ylabel('Scores')
plt.title('Quel modèle fait rire le mieux ?')
plt.xticks(x, models)
plt.legend()
plt.savefig("metrics_comparison.png")
plt.show()

# Graphique 2 : Comparaison des matrices de confusion
plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
sns.heatmap(cm_gnb, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Pas drôle', 'Drôle'], 
            yticklabels=['Pas drôle', 'Drôle'])
plt.title("Matrice - GNB (le comique naïf)")
plt.xlabel("Prédictions")
plt.ylabel("Réel")

plt.subplot(1, 3, 2)
sns.heatmap(cm_xgb, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Pas drôle', 'Drôle'], 
            yticklabels=['Pas drôle', 'Drôle'])
plt.title("Matrice - XGBoost (le pro du rire)")
plt.xlabel("Prédictions")
plt.ylabel("Réel")

plt.subplot(1, 3, 3)
sns.heatmap(cm_lr, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Pas drôle', 'Drôle'], 
            yticklabels=['Pas drôle', 'Drôle'])
plt.title("Matrice - LogReg (le classique marrant)")
plt.xlabel("Prédictions")
plt.ylabel("Réel")
plt.tight_layout()
plt.savefig("confusion_matrices_comparison.png")
plt.show()

# Graphique 3 : Courbe ROC superposée
plt.figure(figsize=(8, 6))
fpr_gnb, tpr_gnb, _ = roc_curve(y_test, y_pred_proba_gnb)
plt.plot(fpr_gnb, tpr_gnb, label=f'GNB (AUC = {auc_gnb:.2f})', color='blue')
fpr_xgb, tpr_xgb, _ = roc_curve(y_test, y_pred_proba_xgb)
plt.plot(fpr_xgb, tpr_xgb, label=f'XGBoost (AUC = {auc_xgb:.2f})', color='orange')
fpr_lr, tpr_lr, _ = roc_curve(y_test, y_pred_proba_lr)
plt.plot(fpr_lr, tpr_lr, label=f'LogReg (AUC = {auc_lr:.2f})', color='green')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('Faux rires')
plt.ylabel('Vrais rires')
plt.title('Courbe ROC : Qui détecte le mieux l’humour ?')
plt.legend()
plt.savefig("roc_curve_comparison.png")
plt.show()
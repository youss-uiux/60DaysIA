import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier, plot_importance
from sklearn.metrics import (accuracy_score, confusion_matrix, 
                            classification_report, roc_curve, auc, RocCurveDisplay)
from sklearn.preprocessing import StandardScaler

# 1. Chargement du fichier local
# Objectif : Charger le dataset Pima Indians Diabetes depuis un fichier CSV local.
data = pd.read_csv("diabetes.csv")

# Vérification des colonnes pour s'assurer que le dataset est correctement chargé.
print("Colonnes disponibles :", data.columns.tolist())

# 2. Prétraitement des données
# Objectif : Nettoyer les données en remplaçant les valeurs aberrantes (0) par NaN, puis imputer par la médiane.
def clean_data(df):
    # Remplacement des 0 par NaN pour les caractéristiques médicales (0 est biologiquement impossible).
    medical_features = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    df[medical_features] = df[medical_features].replace(0, np.nan)
    
    # Imputation par la médiane pour gérer les valeurs manquantes.
    df.fillna(df.median(), inplace=True)
    
    return df

data = clean_data(data)

# 3. Visualisations avant modélisation
# Objectif : Explorer les données avec différents graphiques pour mieux comprendre les relations entre variables.
# Chaque graphique est dans une fenêtre séparée et sauvegardé localement.

# Distribution du Glucose par statut diabétique
plt.figure(figsize=(8, 6))
sns.histplot(data=data, x='Glucose', hue='Outcome', bins=30, kde=True)
plt.title('Distribution du Glucose par statut diabétique')
plt.savefig("glucose_distribution.png")
plt.show()

# Relation Âge/IMC
plt.figure(figsize=(8, 6))
sns.scatterplot(data=data, x='Age', y='BMI', hue='Outcome', palette='viridis')
plt.title('Relation Âge/IMC')
plt.savefig("age_bmi_relation.png")
plt.show()

# Heatmap de corrélation
plt.figure(figsize=(8, 6))
corr_matrix = data.corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Matrice de corrélation')
plt.savefig("correlation_matrix.png")
plt.show()

# Distribution de l'insuline
plt.figure(figsize=(8, 6))
sns.boxplot(x='Outcome', y='Insulin', data=data)
plt.title('Niveau d\'insuline par statut diabétique')
plt.savefig("insulin_distribution.png")
plt.show()

# Relation grossesses/diabète
plt.figure(figsize=(8, 6))
sns.countplot(data=data, x='Pregnancies', hue='Outcome')
plt.title('Nombre de grossesses et diabète')
plt.savefig("pregnancies_diabetes.png")
plt.show()

# 4. Préparation des données
# Objectif : Séparer les features (X) et la cible (y), normaliser les données, et diviser en ensembles d'entraînement/test.
X = data.drop('Outcome', axis=1)
y = data['Outcome']

# Normalisation des features avec StandardScaler pour que toutes les variables aient une échelle comparable.
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Division en ensembles d'entraînement (70%) et de test (30%).
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# 5. Modèle XGBoost optimisé
# Objectif : Entraîner un modèle XGBoost avec des hyperparamètres optimisés pour la classification binaire.
params = {
    'learning_rate': 0.05,        # Taux d'apprentissage pour ajuster la contribution de chaque arbre.
    'n_estimators': 200,          # Nombre d'arbres dans le modèle.
    'max_depth': 4,               # Profondeur maximale des arbres pour éviter le surajustement.
    'subsample': 0.9,             # Fraction des échantillons utilisés pour entraîner chaque arbre.
    'colsample_bytree': 0.7,      # Fraction des features utilisées pour chaque arbre.
    'gamma': 0.1,                 # Régularisation pour limiter la croissance des arbres.
    'eval_metric': 'logloss'      # Métrique d'évaluation (log loss pour la classification binaire).
}

model = XGBClassifier(**params)
model.fit(X_train, y_train)

# 6. Évaluation
# Objectif : Faire des prédictions et évaluer le modèle avec des métriques comme le rapport de classification et la matrice de confusion.
y_pred = model.predict(X_test)
y_probs = model.predict_proba(X_test)[:, 1]

print("\n🔍 Rapport de classification :")
print(classification_report(y_test, y_pred))

print("\n📊 Matrice de confusion :")
print(confusion_matrix(y_test, y_pred))

# 7. Visualisation des résultats
# Objectif : Générer des graphiques pour analyser la performance du modèle (importance des features et courbe ROC).
# Chaque graphique est dans une fenêtre séparée et sauvegardé localement.

# Importance des caractéristiques
plt.figure(figsize=(8, 6))
plot_importance(model, height=0.8)
plt.title('Importance des caractéristiques')
plt.savefig("feature_importance.png")
plt.show()

# Courbe ROC
plt.figure(figsize=(8, 6))
RocCurveDisplay.from_estimator(model, X_test, y_test)
plt.title('Courbe ROC')
plt.plot([0, 1], [0, 1], 'k--')
plt.savefig("roc_curve.png")
plt.show()

# Matrice de confusion (ajoutée pour comparer les données réelles et prédites, comme demandé)
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Non diabétique', 'Diabétique'], 
            yticklabels=['Non diabétique', 'Diabétique'])
plt.title("Matrice de confusion : Réelles vs Prédites (Diabète)")
plt.xlabel("Prédictions")
plt.ylabel("Valeurs réelles")
plt.savefig("confusion_matrix_diabetes.png")
plt.show()

# 8. Interprétation avec SHAP (optionnel mais recommandé)
# Objectif : Analyser l'impact des features sur les prédictions avec SHAP (si installé).
try:
    import shap
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_train)
    
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_train, feature_names=data.columns[:-1])
    plt.title('Analyse SHAP - Impact des caractéristiques')
    plt.savefig("shap_summary.png")
    plt.show()
except ImportError:
    print("\n⚠️ Pour l'analyse SHAP, installez le package : pip install shap")

# 9. Prédiction pour un nouveau patient (exemple)
# Objectif : Faire une prédiction pour un patient fictif pour illustrer l'utilisation pratique du modèle.
new_patient = [[1, 150, 72, 35, 0, 33.6, 0.627, 50]]  # Exemple de données
scaled_patient = scaler.transform(new_patient)
prediction = model.predict(scaled_patient)
probability = model.predict_proba(scaled_patient)[0][1]

print(f"\n🩺 Résultat prédictif pour le nouveau patient :")
print(f"- Classe prédite : {'Diabétique' if prediction[0] else 'Non diabétique'}")
print(f"- Probabilité de diabète : {probability*100:.1f}%")
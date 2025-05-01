import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier, plot_importance

# Charger le dataset
df = pd.read_csv("diabetes.csv")
# Remplacer les 0 par la médiane pour certaines colonnes
cols_with_zeros = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
for col in cols_with_zeros:
    df[col] = df[col].replace(0, df[col].median())
# Séparer features et cible
X = df.drop('Outcome', axis=1)
y = df['Outcome']
# Diviser en train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Calculer scale_pos_weight
scale_pos_weight = (len(y_train) - sum(y_train)) / sum(y_train)
# Entraîner XGBoost
model = XGBClassifier(scale_pos_weight=scale_pos_weight, random_state=42)
model.fit(X_train, y_train)

# Générer le graphique d'importance des features
plt.figure(figsize=(10, 6))
plot_importance(model, importance_type='gain', max_num_features=8, title="Importance des features pour prédire le diabète")
plt.show()
# Optionnel : sauvegarder le graphique pour LinkedIn
plt.savefig("feature_importance_diabetes.png")
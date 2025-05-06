import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import recall_score, precision_score, roc_auc_score, accuracy_score, confusion_matrix, roc_curve

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

# Télécharger les données du S&P 500
sp500 = yf.download('^GSPC', start='2020-01-01', end='2025-01-01')  # Ajuste les dates selon tes besoins
data = sp500[['Open', 'High', 'Low', 'Close', 'Volume']].copy()

# Créer une cible : 1 si la clôture augmente demain, 0 sinon
data['Target'] = np.where(data['Close'].shift(-1) > data['Close'], 1, 0)
data = data.dropna()

# Features et cible
X = data[['Open', 'High', 'Low', 'Close', 'Volume']]
y = data['Target']

# Diviser en train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Standardiser les features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Construire le modèle MLP
model = Sequential([
    Dense(32, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.3),
    Dense(16, activation='relu'),
    Dropout(0.2),
    Dense(8, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Compiler le modèle
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Entraîner le modèle
history = model.fit(X_train, y_train, validation_split=0.2, epochs=20, batch_size=32, verbose=1)

# Prédictions avec seuil par défaut (0.5)
y_pred_proba = model.predict(X_test)
y_pred = (y_pred_proba >= 0.5).astype(int).flatten()

# Métriques
recall = recall_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred_proba)
accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

# Afficher les résultats
print("Métriques sur l'ensemble de test :")
print(f"Rappel = {recall:.2f}, Précision = {precision:.2f}, AUC = {auc:.2f}, Exactitude = {accuracy:.2f}")
print("Matrice de confusion :")
print(cm)

# Graphique : Matrice de confusion
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Baisse', 'Hausse'], 
            yticklabels=['Baisse', 'Hausse'])
plt.title("Matrice de Confusion - Prédiction des Tendances (MLP)")
plt.xlabel("Prédictions")
plt.ylabel("Réel")
plt.savefig("confusion_matrix_sp500.png")
plt.show()

# Graphique : Courbe ROC
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
plt.figure
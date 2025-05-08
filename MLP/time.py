import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import recall_score, precision_score, roc_auc_score, accuracy_score, confusion_matrix, roc_curve
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

# Télécharger les données du S&P 500
sp500 = yf.download('^GSPC', start='2020-01-01', end='2025-01-01')  # Ajuste les dates si besoin
data = sp500[['Open', 'High', 'Low', 'Close', 'Volume']].copy()

# Créer une cible : 1 si la clôture augmente demain, 0 sinon
data['Target'] = np.where(data['Close'].shift(-1) > data['Close'], 1, 0)
data = data.dropna()

# Préparer les données comme une série temporelle
def create_sequences(X, y, time_steps=5):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        Xs.append(X[i:(i + time_steps)])
        ys.append(y[i + time_steps])
    return np.array(Xs), np.array(ys)

# Features et cible
X = data[['Open', 'High', 'Low', 'Close', 'Volume']].values
y = data['Target'].values

# Créer des séquences temporelles
time_steps = 5
X_seq, y_seq = create_sequences(X, y, time_steps)

# Diviser en train/test
train_size = int(len(X_seq) * 0.8)
X_train, X_test = X_seq[:train_size], X_seq[train_size:]
y_train, y_test = y_seq[:train_size], y_seq[train_size:]

# Standardiser les features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
X_test = scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)

# Construire le modèle MLP
model = Sequential([
    Dense(32, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])),
    Dropout(0.3),
    Dense(16, activation='relu'),
    Dropout(0.2),
    Dense(8, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Compiler le modèle
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Entraîner le modèle
history = model.fit(X_train, y_train, validation_split=0.2, epochs=30, batch_size=32, verbose=1)

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
plt.savefig("confusion_matrix_sp500_time_series.png")
plt.show()

# Graphique : Courbe ROC
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, label=f'MLP (AUC = {auc:.2f})', color='green')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('Taux de faux positifs')
plt.ylabel('Taux de vrais positifs')
plt.title('Courbe ROC - Prédiction des Tendances (MLP)')
plt.legend()
plt.savefig("roc_curve_sp500_time_series.png")
plt.show()
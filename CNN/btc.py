import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

# 1. Charger les données
# Remplacez 'BTC-USD.csv' par le nom de votre fichier CSV
try:
    df = pd.read_csv('BTC-USD.csv')
except FileNotFoundError:
    print("Fichier 'BTC-USD.csv' introuvable. Veuillez fournir le fichier ou utiliser des données synthétiques.")
    exit(1)

# 2. Préparer les données
# Convertir la colonne 'Date' en format datetime
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date')
df.set_index('Date', inplace=True)

# Supprimer les valeurs manquantes
df = df.dropna(subset=['Close', 'Volume'])

# Sélectionner les variables (multivarié : Close et Volume)
data = df[['Close', 'Volume']].values

# 3. Normaliser les données
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

# 4. Créer des séquences pour l'entraînement
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length, 0])  # Prédire uniquement 'Close'
    return np.array(X), np.array(y)

# Hyperparamètres optimisés
sequence_length = 60  # 60 jours pour capturer des tendances à long terme
train_test_split = 0.8
filters = 128  # Plus de filtres pour une meilleure capacité
kernel_size = 3  # Noyau plus petit pour des motifs fins
pool_size = 2
num_epochs = 100  # Plus d'époques pour un meilleur entraînement
batch_size = 16  # Batch plus grand pour une convergence stable

X, y = create_sequences(scaled_data, sequence_length)

# 5. Diviser les données
train_size = int(len(X) * train_test_split)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Reshape pour TCN (multivarié : 2 features)
X_train = X_train.reshape(-1, sequence_length, 2)
X_test = X_test.reshape(-1, sequence_length, 2)

# 6. Créer le modèle TCN
model = Sequential()
model.add(Conv1D(filters=filters, kernel_size=kernel_size, activation='relu', input_shape=(sequence_length, 2)))
model.add(MaxPooling1D(pool_size=pool_size))
model.add(Conv1D(filters=filters // 2, kernel_size=kernel_size, activation='relu'))
model.add(MaxPooling1D(pool_size=pool_size))
model.add(Flatten())
model.add(Dense(units=100, activation='relu'))
model.add(Dense(units=1))

# Compiler le modèle avec un taux d'apprentissage réduit
model.compile(loss='mean_squared_error', optimizer=Adam(learning_rate=0.0001))

# 7. Entraîner le modèle
history = model.fit(X_train, y_train, epochs=num_epochs, batch_size=batch_size, validation_split=0.1, verbose=1)

# 8. Faire des prédictions
predictions = model.predict(X_test)

# Inverser la normalisation pour les prédictions et les valeurs réelles
# Créer un tableau temporaire pour l'inversion (seulement pour 'Close')
predictions_full = np.zeros((predictions.shape[0], 2))
predictions_full[:, 0] = predictions[:, 0]
y_test_full = np.zeros((y_test.shape[0], 2))
y_test_full[:, 0] = y_test

predictions = scaler.inverse_transform(predictions_full)[:, 0]
y_test_scaled = scaler.inverse_transform(y_test_full)[:, 0]

# 9. Calculer les métriques
rmse = np.sqrt(mean_squared_error(y_test_scaled, predictions))
mae = mean_absolute_error(y_test_scaled, predictions)
r2 = r2_score(y_test_scaled, predictions)

print(f'RMSE: {rmse:.2f} USD')
print(f'MAE: {mae:.2f} USD')
print(f'R²: {r2:.2f}')

# 10. Visualiser les résultats
plt.figure(figsize=(14, 5))
plt.plot(y_test_scaled, color='blue', label='Prix réel du Bitcoin (USD)')
plt.plot(predictions, color='red', label='Prédictions TCN')
plt.title('Prédiction des prix du Bitcoin avec TCN pour le contexte africain')
plt.xlabel('Temps')
plt.ylabel('Prix (USD)')
plt.legend()
plt.savefig('btc_predictions.png')  # Sauvegarder pour LinkedIn
plt.show()

# 11. Visualiser la perte
plt.figure(figsize=(14, 5))
plt.plot(history.history['loss'], label='Perte (entraînement)')
plt.plot(history.history['val_loss'], label='Perte (validation)')
plt.title('Perte du modèle TCN')
plt.xlabel('Époque')
plt.ylabel('Perte (MSE)')
plt.legend()
plt.show()
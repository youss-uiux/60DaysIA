import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Flatten
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from ta.momentum import RSIIndicator

# 1. Charger les données
try:
    df = pd.read_csv('BTC-USD.csv')
except FileNotFoundError:
    print("Fichier 'BTC-USD.csv' introuvable. Veuillez fournir le fichier.")
    exit(1)

# 2. Préparer les données
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date')
df.set_index('Date', inplace=True)

# Calculer les rendements logarithmiques pour réduire la volatilité
df['Log_Return'] = np.log(df['Close'] / df['Close'].shift(1))

# Ajouter des indicateurs techniques
df['RSI'] = RSIIndicator(df['Close']).rsi()
df['MA7'] = df['Close'].rolling(window=7).mean()

# Supprimer les valeurs manquantes
df = df.dropna()

# Sélectionner les features (Log_Return, Volume, RSI, MA7)
data = df[['Log_Return', 'Volume', 'RSI', 'MA7']].values

# 3. Normaliser les données
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

# 4. Créer des séquences
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length, 0])  # Prédire Log_Return
    return np.array(X), np.array(y)

# Hyperparamètres optimisés
sequence_length = 90  # 90 jours pour capturer des tendances long terme
train_test_split = 0.8
filters = 128
kernel_size = 3
num_epochs = 150
batch_size = 16

X, y = create_sequences(scaled_data, sequence_length)

# 5. Diviser les données
train_size = int(len(X) * train_test_split)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Reshape pour le modèle (4 features)
X_train = X_train.reshape(-1, sequence_length, 4)
X_test = X_test.reshape(-1, sequence_length, 4)

# 6. Créer le modèle TCN + LSTM
model = Sequential()
model.add(Conv1D(filters=filters, kernel_size=kernel_size, activation='relu', input_shape=(sequence_length, 4)))
model.add(MaxPooling1D(pool_size=2))
model.add(Conv1D(filters=filters // 2, kernel_size=kernel_size, activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(LSTM(50, return_sequences=False))
model.add(Dense(units=100, activation='relu'))
model.add(Dense(units=1))

# Compiler avec un taux d'apprentissage réduit
model.compile(loss='mean_squared_error', optimizer=Adam(learning_rate=0.0001))

# 7. Entraîner le modèle
history = model.fit(X_train, y_train, epochs=num_epochs, batch_size=batch_size, validation_split=0.1, verbose=1)

# 8. Faire des prédictions
predictions = model.predict(X_test)

# Inverser la normalisation
predictions_full = np.zeros((predictions.shape[0], 4))
predictions_full[:, 0] = predictions[:, 0]
y_test_full = np.zeros((y_test.shape[0], 4))
y_test_full[:, 0] = y_test

predictions = scaler.inverse_transform(predictions_full)[:, 0]
y_test_scaled = scaler.inverse_transform(y_test_full)[:, 0]

# Convertir les rendements logarithmiques en prix pour évaluation
# (Approximation : utiliser les prix réels pour contextualiser)
last_prices = df['Close'].values[-len(y_test_scaled):]
predicted_prices = last_prices * np.exp(predictions)
actual_prices = last_prices * np.exp(y_test_scaled)

# 9. Calculer les métriques
rmse = np.sqrt(mean_squared_error(actual_prices, predicted_prices))
mae = mean_absolute_error(actual_prices, predicted_prices)
r2 = r2_score(actual_prices, predicted_prices)

print(f'RMSE: {rmse:.2f} USD')
print(f'MAE: {mae:.2f} USD')
print(f'R²: {r2:.2f}')

# 10. Visualiser les résultats
plt.figure(figsize=(14, 5))
plt.plot(actual_prices, color='blue', label='Prix réel du Bitcoin (USD)')
plt.plot(predicted_prices, color='red', label='Prédictions TCN+LSTM')
plt.title('Prédiction des prix du Bitcoin avec TCN+LSTM')
plt.xlabel('Temps')
plt.ylabel('Prix (USD)')
plt.legend()
plt.savefig('btc_predictions.png')
plt.show()

# 11. Visualiser la perte
plt.figure(figsize=(14, 5))
plt.plot(history.history['loss'], label='Perte (entraînement)')
plt.plot(history.history['val_loss'], label='Perte (validation)')
plt.title('Perte du modèle TCN+LSTM')
plt.xlabel('Époque')
plt.ylabel('Perte (MSE)')
plt.legend()
plt.show()
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import tensorflow as tf

# Fixer les graines pour reproductibilité
np.random.seed(42)
tf.random.set_seed(42)

# Charger les données
df = pd.read_csv('BTC-USD.csv')
data = df['Close'].values.reshape(-1, 1)

# Normalisation
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)

# Création des séquences
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)

sequence_length = 60
X, y = create_sequences(scaled_data, sequence_length)

# Séparation train/test
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Construction du modèle LSTM
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(sequence_length, 1)),
    Dropout(0.2),
    LSTM(50),
    Dropout(0.2),
    Dense(25),
    Dense(1)
])
model.compile(optimizer='adam', loss='mean_squared_error')

# Entraînement
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.1, verbose=1)

# Prédictions
predictions = model.predict(X_test)
y_pred = scaler.inverse_transform(predictions)
y_true = scaler.inverse_transform(y_test)

# Fonctions de métriques
def evaluate_model(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    r2 = r2_score(y_true, y_pred)
    corr = np.corrcoef(y_true.flatten(), y_pred.flatten())[0, 1]
    bias = np.mean(y_pred - y_true)
    threshold = 0.1 * np.mean(y_true)
    accuracy = np.mean(np.abs(y_true - y_pred) < threshold) * 100
    return rmse, mae, mape, r2, corr, bias, accuracy

rmse, mae, mape, r2, corr, bias, accuracy = evaluate_model(y_true, y_pred)

# Affichage des métriques
print("\n📊 **MÉTRIQUES DU MODÈLE LSTM SUR BTC**")
print(f"✔️ RMSE:  {rmse:.2f} USD")
print(f"✔️ MAE:   {mae:.2f} USD")
print(f"✔️ MAPE:  {mape:.2f} %")
print(f"✔️ R²:    {r2:.4f}")
print(f"✔️ Corrélation: {corr:.4f}")
print(f"✔️ Biais: {bias:.2f} USD")
print(f"✔️ Précision (<10% erreur): {accuracy:.2f} %")

# Visualisation : Prédiction vs Réel
plt.figure(figsize=(14, 5))
plt.plot(y_true, label='Prix réel', color='blue')
plt.plot(y_pred, label='Prédiction', color='red', linestyle='--')
plt.title('Prédiction des prix de Bitcoin avec LSTM')
plt.xlabel('Temps')
plt.ylabel('Prix (USD)')
plt.legend()
plt.show()

# Visualisation : courbes de perte
plt.figure(figsize=(14, 5))
plt.plot(history.history['loss'], label='Perte entraînement')
plt.plot(history.history['val_loss'], label='Perte validation')
plt.title('Courbes de perte du modèle')
plt.xlabel('Époques')
plt.ylabel('MSE')
plt.legend()
plt.show()

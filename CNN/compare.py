import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Flatten, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import plot_model
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

# Calculer les rendements logarithmiques
df['Log_Return'] = np.log(df['Close'] / df['Close'].shift(1))

# Ajouter des indicateurs techniques
df['RSI'] = RSIIndicator(df['Close']).rsi()
df['MA7'] = df['Close'].rolling(window=7).mean()

# Supprimer les valeurs manquantes
df = df.dropna()

# Sélectionner les features
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

# Hyperparamètres
sequence_length = 90
train_test_split = 0.8
filters = 128
kernel_size = 3
num_epochs = 100
batch_size = 16

X, y = create_sequences(scaled_data, sequence_length)

# 5. Diviser les données
train_size = int(len(X) * train_test_split)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Reshape pour le modèle (4 features)
X_train = X_train.reshape(-1, sequence_length, 4)
X_test = X_test.reshape(-1, sequence_length, 4)

# 6. Modèle 1 : TCN simple
tcn_model = Sequential(name='TCN_Model')
tcn_model.add(Input(shape=(sequence_length, 4), name='Input'))
tcn_model.add(Conv1D(filters=filters, kernel_size=kernel_size, activation='relu', name='Conv1D_1'))
tcn_model.add(MaxPooling1D(pool_size=2, name='MaxPooling_1'))
tcn_model.add(Conv1D(filters=filters // 2, kernel_size=kernel_size, activation='relu', name='Conv1D_2'))
tcn_model.add(MaxPooling1D(pool_size=2, name='MaxPooling_2'))
tcn_model.add(Flatten(name='Flatten'))
tcn_model.add(Dense(units=100, activation='relu', name='Dense_1'))
tcn_model.add(Dropout(0.2, name='Dropout_1'))
tcn_model.add(Dense(units=1, name='Output'))

tcn_model.compile(loss='mean_squared_error', optimizer=Adam(learning_rate=0.0001))

# Visualiser le schéma du TCN
try:
    plot_model(tcn_model, to_file='tcn_model.png', show_shapes=True, show_layer_names=True)
    print("Schéma du TCN sauvegardé : tcn_model.png")
except ImportError:
    print("Erreur : Impossible de générer le schéma du TCN. Assurez-vous que pydot et graphviz sont installés.")

# 7. Modèle 2 : TCN + LSTM
tcn_lstm_model = Sequential(name='TCN_LSTM_Model')
tcn_lstm_model.add(Input(shape=(sequence_length, 4), name='Input'))
tcn_lstm_model.add(Conv1D(filters=filters, kernel_size=kernel_size, activation='relu', name='Conv1D_1'))
tcn_lstm_model.add(MaxPooling1D(pool_size=2, name='MaxPooling_1'))
tcn_lstm_model.add(Conv1D(filters=filters // 2, kernel_size=kernel_size, activation='relu', name='Conv1D_2'))
tcn_lstm_model.add(MaxPooling1D(pool_size=2, name='MaxPooling_2'))
tcn_lstm_model.add(LSTM(50, return_sequences=False, name='LSTM'))
tcn_lstm_model.add(Dropout(0.2, name='Dropout_1'))
tcn_lstm_model.add(Dense(units=100, activation='relu', name='Dense_1'))
tcn_lstm_model.add(Dropout(0.2, name='Dropout_2'))
tcn_lstm_model.add(Dense(units=1, name='Output'))

tcn_lstm_model.compile(loss='mean_squared_error', optimizer=Adam(learning_rate=0.0001))

# Visualiser le schéma du TCN+LSTM
try:
    plot_model(tcn_lstm_model, to_file='tcn_lstm_model.png', show_shapes=True, show_layer_names=True)
    print("Schéma du TCN+LSTM sauvegardé : tcn_lstm_model.png")
except ImportError:
    print("Erreur : Impossible de générer le schéma du TCN+LSTM. Assurez-vous que pydot et graphviz sont installés.")

# 8. Entraîner les modèles
print("Entraînement du TCN simple...")
tcn_history = tcn_model.fit(X_train, y_train, epochs=num_epochs, batch_size=batch_size, validation_split=0.1, verbose=1)

print("\nEntraînement du TCN+LSTM...")
tcn_lstm_history = tcn_lstm_model.fit(X_train, y_train, epochs=num_epochs, batch_size=batch_size, validation_split=0.1, verbose=1)

# 9. Faire des prédictions
tcn_predictions = tcn_model.predict(X_test)
tcn_lstm_predictions = tcn_lstm_model.predict(X_test)

# Inverser la normalisation
predictions_full = np.zeros((tcn_predictions.shape[0], 4))
predictions_full[:, 0] = tcn_predictions[:, 0]
tcn_predictions = scaler.inverse_transform(predictions_full)[:, 0]

predictions_full[:, 0] = tcn_lstm_predictions[:, 0]
tcn_lstm_predictions = scaler.inverse_transform(predictions_full)[:, 0]

y_test_full = np.zeros((y_test.shape[0], 4))
y_test_full[:, 0] = y_test
y_test_scaled = scaler.inverse_transform(y_test_full)[:, 0]

# Convertir les rendements logarithmiques en prix
last_prices = df['Close'].values[-len(y_test_scaled):]
tcn_predicted_prices = last_prices * np.exp(tcn_predictions)
tcn_lstm_predicted_prices = last_prices * np.exp(tcn_lstm_predictions)
actual_prices = last_prices * np.exp(y_test_scaled)

# 10. Calculer les métriques
tcn_rmse = np.sqrt(mean_squared_error(actual_prices, tcn_predicted_prices))
tcn_mae = mean_absolute_error(actual_prices, tcn_predicted_prices)
tcn_r2 = r2_score(actual_prices, tcn_predicted_prices)

tcn_lstm_rmse = np.sqrt(mean_squared_error(actual_prices, tcn_lstm_predicted_prices))
tcn_lstm_mae = mean_absolute_error(actual_prices, tcn_lstm_predicted_prices)
tcn_lstm_r2 = r2_score(actual_prices, tcn_lstm_predicted_prices)

print("\nMétriques pour TCN simple :")
print(f'RMSE: {tcn_rmse:.2f} USD')
print(f'MAE: {tcn_mae:.2f} USD')
print(f'R²: {tcn_r2:.2f}')

print("\nMétriques pour TCN+LSTM :")
print(f'RMSE: {tcn_lstm_rmse:.2f} USD')
print(f'MAE: {tcn_lstm_mae:.2f} USD')
print(f'R²: {tcn_lstm_r2:.2f}')

# 11. Visualiser les prédictions
plt.figure(figsize=(14, 5))
plt.plot(actual_prices, color='blue', label='Prix réel du Bitcoin (USD)')
plt.plot(tcn_predicted_prices, color='red', label='Prédictions TCN')
plt.plot(tcn_lstm_predicted_prices, color='green', label='Prédictions TCN+LSTM')
plt.title('Comparaison des prédictions : TCN vs TCN+LSTM')
plt.xlabel('Temps')
plt.ylabel('Prix (USD)')
plt.legend()
plt.savefig('btc_predictions_comparison.png')
plt.show()

# 12. Visualiser les pertes
plt.figure(figsize=(14, 5))
plt.plot(tcn_history.history['loss'], color='red', label='Perte TCN (entraînement)')
plt.plot(tcn_history.history['val_loss'], color='pink', label='Perte TCN (validation)')
plt.plot(tcn_lstm_history.history['loss'], color='green', label='Perte TCN+LSTM (entraînement)')
plt.plot(tcn_lstm_history.history['val_loss'], color='lightgreen', label='Perte TCN+LSTM (validation)')
plt.title('Comparaison des pertes : TCN vs TCN+LSTM')
plt.xlabel('Époque')
plt.ylabel('Perte (MSE)')
plt.legend()
plt.show()
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

# 1. Charger les données
# Remplacez 'sales_data.csv' par le nom de votre fichier CSV
try:
    df = pd.read_csv('sales_data.csv')
except FileNotFoundError:
    print("Fichier 'sales_data.csv' introuvable. Veuillez fournir le fichier ou utiliser des données synthétiques.")
    exit(1)

# 2. Préparer les données
# Convertir la colonne 'date' en format datetime
df['date'] = pd.to_datetime(df['date'], format='%d.%m.%Y')

# Agréger les ventes quotidiennes totales (somme de 'item_cnt_day' par date)
daily_sales = df.groupby('date')['item_cnt_day'].sum().reset_index()
daily_sales = daily_sales.sort_values('date')

# Créer la série temporelle
data = daily_sales['item_cnt_day'].values.reshape(-1, 1)

# 3. Normaliser les données
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

# 4. Créer des séquences pour l'entraînement
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)

# Hyperparamètres
sequence_length = 30  # 30 jours pour capturer les tendances
train_test_split = 0.8
filters = 64
kernel_size = 4
pool_size = 2
num_epochs = 50
batch_size = 8

X, y = create_sequences(scaled_data, sequence_length)

# 5. Diviser les données
train_size = int(len(X) * train_test_split)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Reshape pour TCN
X_train = X_train.reshape(-1, sequence_length, 1)
X_test = X_test.reshape(-1, sequence_length, 1)

# 6. Créer le modèle TCN
model = Sequential()
model.add(Conv1D(filters=filters, kernel_size=kernel_size, activation='relu', input_shape=(sequence_length, 1)))
model.add(MaxPooling1D(pool_size=pool_size))
model.add(Flatten())
model.add(Dense(units=50, activation='relu'))
model.add(Dense(units=1))

# Compiler le modèle
model.compile(loss='mean_squared_error', optimizer='adam')

# 7. Entraîner le modèle
history = model.fit(X_train, y_train, epochs=num_epochs, batch_size=batch_size, validation_split=0.1, verbose=1)

# 8. Faire des prédictions
predictions = model.predict(X_test)
predictions = scaler.inverse_transform(predictions)
y_test_scaled = scaler.inverse_transform(y_test)

# 9. Calculer les métriques
rmse = np.sqrt(mean_squared_error(y_test_scaled, predictions))
print(f'RMSE: {rmse:.2f} unités vendues')

# 10. Visualiser les résultats
plt.figure(figsize=(14, 5))
plt.plot(y_test_scaled, color='blue', label='Ventes quotidiennes réelles')
plt.plot(predictions, color='red', label='Prédictions TCN')
plt.title('Prédiction des ventes quotidiennes dans un marché ouest-africain avec TCN')
plt.xlabel('Temps')
plt.ylabel('Unités vendues')
plt.legend()
plt.savefig('sales_predictions.png')  # Sauvegarder pour LinkedIn
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
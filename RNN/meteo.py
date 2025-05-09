import pandas as pd
import numpy as np
import requests
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense
import matplotlib.pyplot as plt

# 1. Récupérer les données météo
url = (
    "https://archive-api.open-meteo.com/v1/archive"
    "?latitude=13.5128&longitude=2.1128"
    "&start_date=2025-02-01&end_date=2025-04-30"
    "&daily=temperature_2m_max,temperature_2m_min,precipitation_sum"
    "&timezone=Africa%2FLagos"
)
data = requests.get(url).json()
df = pd.DataFrame(data["daily"])
df["date"] = pd.to_datetime(df["time"])
df.set_index("date", inplace=True)
df.drop(columns=["time"], inplace=True)

# 2. Normalisation
scaler = StandardScaler()
scaled = scaler.fit_transform(df)

# 3. Créer les séquences pour RNN
def create_sequences(data, n_lags):
    X, y = [], []
    for i in range(len(data) - n_lags):
        X.append(data[i:i + n_lags])
        y.append(data[i + n_lags])
    return np.array(X), np.array(y)

n_lags = 7  # une semaine d'historique pour prédire le lendemain
X, y = create_sequences(scaled, n_lags)

# Reshape pour RNN
X = X.reshape((X.shape[0], X.shape[1], X.shape[2]))  # [samples, time, features]

# 4. Split train/test
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# 5. Modèle RNN simple
model = Sequential()
model.add(SimpleRNN(50, activation='tanh', input_shape=(n_lags, X.shape[2])))
model.add(Dense(X.shape[2]))  # prédire toutes les variables météo
model.compile(optimizer='adam', loss='mse')
model.fit(X_train, y_train, epochs=50, batch_size=16, validation_split=0.1, verbose=1)

# 6. Prédictions
y_pred = model.predict(X_test)
y_pred_rescaled = scaler.inverse_transform(y_pred)
y_test_rescaled = scaler.inverse_transform(y_test)

# 7. Affichage
plt.figure(figsize=(12, 5))
plt.plot(y_test_rescaled[:, 0], label="Température max réelle")
plt.plot(y_pred_rescaled[:, 0], label="Température max prédite")
plt.legend()
plt.title("Prédiction de la température maximale à Niamey (RNN)")
plt.xlabel("Jours")
plt.ylabel("Température (°C)")
plt.grid(True)
plt.show()

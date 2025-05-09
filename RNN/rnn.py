import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import pandas as pd
import requests
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

# Récupérer les données via Alpha Vantage
api_key = 'YOUR_API_KEY'
url = f'https://www.alphavantage.co/query?function=DIGITAL_CURRENCY_DAILY&symbol=BTC&market=EUR&apikey={api_key}'
try:
    r = requests.get(url)
    data = r.json()
    if 'Time Series (Digital Currency Daily)' not in data:
        raise KeyError("Erreur : Les données historiques ne sont pas disponibles.")
except Exception as e:
    print(f"Erreur : {e}")
    exit()

time_series = data['Time Series (Digital Currency Daily)']
df = pd.DataFrame.from_dict(time_series, orient='index')
df = df.astype(float)

close_column = None
if '4b. close (EUR)' in df.columns:
    close_column = '4b. close (EUR)'
elif '4a. close (EUR)' in df.columns:
    close_column = '4a. close (EUR)'
elif '4. close' in df.columns:
    close_column = '4. close'
else:
    raise KeyError("Aucune colonne de prix de clôture trouvée.")

data = df[close_column].values
data = np.diff(data)

# Hyperparamètres (comme dans l'exemple)
num_lags = 100
train_test_split = 0.80
num_neurons_in_hidden_layers = 20
num_epochs = 500
batch_size = 16

# Prétraitement
def create_sequences(data, num_lags, train_test_split):
    X, y = [], []
    for i in range(len(data) - num_lags - 1):
        X.append(data[i:(i + num_lags)])
        y.append(data[i + num_lags])
    return np.array(X), np.array(y)

X, y = create_sequences(data, num_lags, train_test_split)
train_size = int(len(X) * train_test_split)
x_train, x_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Normalisation
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# Ajuster les dimensions pour le RNN (3D : [samples, timesteps, features])
x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))

# Modèle RNN avec early stopping
model = Sequential()
model.add(SimpleRNN(num_neurons_in_hidden_layers, input_shape=(num_lags, 1), return_sequences=False))
model.add(Dropout(0.3))
model.add(Dense(num_neurons_in_hidden_layers, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')

# Early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)

# Entraîner le modèle
history = model.fit(x_train, y_train.reshape(-1, 1), epochs=num_epochs, batch_size=batch_size, 
                    validation_split=0.2, verbose=1, callbacks=[early_stopping])

# Tracer la courbe de perte
plt.figure(figsize=(12, 6))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss Curve - Bitcoin Returns (BTC/EUR) with RNN')
plt.xlabel('Epoch')
plt.ylabel('Loss Value')
plt.grid(True)
plt.legend()
plt.savefig('bitcoin_eur_returns_rnn_loss.png')
plt.show()

# Prédictions
y_predicted_train = model.predict(x_train)
y_predicted = model.predict(x_test)

# Calculer l'erreur
mse_train = mean_squared_error(y_train, y_predicted_train)
mse_test = mean_squared_error(y_test, y_predicted)
print(f"Mean Squared Error (Train): {mse_train:.6f}")
print(f"Mean Squared Error (Test): {mse_test:.6f}")

# Visualisation
plt.figure(figsize=(12, 6))
plt.plot(y_test, label='Valeurs réelles (Test)')
plt.plot(y_predicted, label='Prédictions (Test)')
plt.title('Prédictions vs Valeurs Réelles - Bitcoin Returns (BTC/EUR) with RNN')
plt.xlabel('Temps')
plt.ylabel('Rendement')
plt.legend()
plt.savefig('bitcoin_eur_returns_rnn.png')
plt.show()
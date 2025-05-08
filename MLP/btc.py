import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
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

# Hyperparamètres
num_lags = 50
train_test_split = 0.80
num_neurons_in_hidden_layers = 40
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

x_train = x_train.reshape((x_train.shape[0], x_train.shape[1]))
x_test = x_test.reshape((x_test.shape[0], x_test.shape[1]))

# Modèle MLP
model = Sequential()
model.add(Dense(num_neurons_in_hidden_layers, input_dim=num_lags, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(num_neurons_in_hidden_layers, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')

# Callback pour la courbe de perte
losses = []
epochs = []
class LossCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        losses.append(logs['loss'])
        epochs.append(epoch + 1)
        plt.clf()
        plt.plot(epochs, losses, marker='o')
        plt.title('Loss Curve - Bitcoin Returns (BTC/EUR)')
        plt.xlabel('Epoch')
        plt.ylabel('Loss Value')
        plt.grid(True)
        plt.pause(0.01)
        plt.savefig('loss.png')
        

# Entraîner le modèle avec la visualisation
model.fit(x_train, y_train.reshape(-1, 1), epochs=num_epochs, batch_size=batch_size, verbose=0, callbacks=[LossCallback()])
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
plt.title('Prédictions vs Valeurs Réelles - Bitcoin Returns (BTC/EUR)')
plt.xlabel('Temps')
plt.ylabel('Rendement')
plt.legend()
plt.savefig('bitcoin_eur_returns_prediction_adjusted.png')
plt.show()
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import pandas as pd
import requests
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler

# Récupérer les données historiques de BTC/EUR via Alpha Vantage
# Remplace 'YOUR_API_KEY' par ta clé API obtenue sur https://www.alphavantage.co/support/#api-key
api_key = '3YZUQJ9CPFR52AQ4'
url = f'https://www.alphavantage.co/query?function=DIGITAL_CURRENCY_DAILY&symbol=BTC&market=EUR&apikey={api_key}'

# Faire la requête à l'API
try:
    r = requests.get(url)
    data = r.json()
    if 'Time Series (Digital Currency Daily)' not in data:
        raise KeyError("Erreur : Les données historiques ne sont pas disponibles. Vérifie ta clé API ou les limites de requêtes.")
except Exception as e:
    print(f"Erreur lors de la récupération des données : {e}")
    exit()

# Extraire les données historiques
time_series = data['Time Series (Digital Currency Daily)']
df = pd.DataFrame.from_dict(time_series, orient='index')
df = df.astype(float)

# Afficher les colonnes disponibles pour inspection
print("Colonnes disponibles dans les données :", df.columns)

# Utiliser la bonne colonne pour le prix de clôture
# On essaie '4b. close (EUR)' ou '4. close' selon ce qui est disponible
close_column = None
if '4b. close (EUR)' in df.columns:
    close_column = '4b. close (EUR)'
elif '4a. close (EUR)' in df.columns:
    close_column = '4a. close (EUR)'
elif '4. close' in df.columns:
    close_column = '4. close'
else:
    raise KeyError("Aucune colonne de prix de clôture trouvée. Vérifie les données renvoyées par l'API.")

data = df[close_column].values

# Différencier les données pour les rendre stationnaires
data = np.diff(data)

# Définir les hyperparamètres
num_lags = 100
train_test_split = 0.80
num_neurons_in_hidden_layers = 40
num_epochs = 1000
batch_size = 16

# Fonction de prétraitement
def create_sequences(data, num_lags, train_test_split):
    X, y = [], []
    for i in range(len(data) - num_lags - 1):
        X.append(data[i:(i + num_lags)])
        y.append(data[i + num_lags])
    X, y = np.array(X), np.array(y)
    train_size = int(len(X) * train_test_split)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    return X_train, y_train, X_test, y_test

# Préparer les données
x_train, y_train, x_test, y_test = create_sequences(data, num_lags, train_test_split)

# Normaliser les données
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# Ajuster les dimensions pour le MLP
x_train = x_train.reshape((x_train.shape[0], x_train.shape[1]))
x_test = x_test.reshape((x_test.shape[0], x_test.shape[1]))

# Construire le modèle MLP
model = Sequential()
model.add(Dense(num_neurons_in_hidden_layers, input_dim=num_lags, activation='relu'))  # Première couche cachée
model.add(Dense(num_neurons_in_hidden_layers, activation='relu'))  # Deuxième couche cachée
model.add(Dense(1))  # Couche de sortie (régression)
model.compile(loss='mean_squared_error', optimizer='adam')

# Entraîner le modèle
history = model.fit(x_train, y_train.reshape(-1, 1), epochs=num_epochs, batch_size=batch_size, verbose=1)

# Prédictions
y_predicted_train = model.predict(x_train)
y_predicted = model.predict(x_test)

# Calculer l'erreur (MSE)
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
plt.savefig('bitcoin_eur_returns_prediction.png')
plt.show()
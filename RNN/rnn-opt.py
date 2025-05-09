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

# Fonction de prétraitement
def create_sequences(data, num_lags, train_test_split):
    X, y = [], []
    for i in range(len(data) - num_lags - 1):
        X.append(data[i:(i + num_lags)])
        y.append(data[i + num_lags])
    return np.array(X), np.array(y)

# Fonction d'optimisation
def optimize_rnn(data, param_grid, train_test_split=0.80):
    best_mse = float('inf')
    best_params = None
    all_predictions = []
    
    for lags in param_grid['num_lags']:
        for units in param_grid['num_units']:
            for epochs in param_grid['num_epochs']:
                for batch in param_grid['batch_size']:
                    # Préparer les données
                    X, y = create_sequences(data, lags, train_test_split)
                    train_size = int(len(X) * train_test_split)
                    x_train, x_test = X[:train_size], X[train_size:]
                    y_train, y_test = y[:train_size], y[train_size:]

                    # Normalisation
                    scaler = StandardScaler()
                    x_train = scaler.fit_transform(x_train)
                    x_test = scaler.transform(x_test)

                    # Ajuster les dimensions pour le RNN
                    x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
                    x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))

                    # Modèle RNN
                    model = Sequential()
                    model.add(SimpleRNN(units, input_shape=(lags, 1), return_sequences=False))
                    model.add(Dropout(0.3))
                    model.add(Dense(20, activation='relu'))
                    model.add(Dropout(0.3))
                    model.add(Dense(1))
                    model.compile(loss='mean_squared_error', optimizer='adam')

                    # Early stopping
                    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

                    # Entraîner le modèle
                    history = model.fit(x_train, y_train.reshape(-1, 1), epochs=epochs, batch_size=batch,
                                      validation_split=0.2, verbose=0, callbacks=[early_stopping])

                    # Prédictions
                    y_predicted_train = model.predict(x_train, verbose=0)
                    y_predicted = model.predict(x_test, verbose=0)

                    # Calculer l'erreur
                    mse_train = mean_squared_error(y_train, y_predicted_train)
                    mse_test = mean_squared_error(y_test, y_predicted)
                    print(f"Lags: {lags}, Units: {units}, Epochs: {epochs}, Batch: {batch}")
                    print(f"MSE Train: {mse_train:.6f}, MSE Test: {mse_test:.6f}")

                    # Mettre à jour le meilleur modèle
                    if mse_test < best_mse:
                        best_mse = mse_test
                        best_params = {'num_lags': lags, 'num_units': units, 'num_epochs': epochs, 'batch_size': batch}

                    # Stocker les prédictions pour l'ensembling
                    all_predictions.append(y_predicted)

    # Ensembling : moyenne des prédictions
    ensemble_pred = np.mean(all_predictions, axis=0)
    ensemble_mse = mean_squared_error(y_test, ensemble_pred)
    print(f"\nMeilleur MSE Test: {best_mse:.6f} avec {best_params}")
    print(f"MSE Test Ensembling: {ensemble_mse:.6f}")

    return best_params, ensemble_pred, y_test

# Grille d'hyperparamètres à tester
param_grid = {
    'num_lags': [50, 100],
    'num_units': [20, 50],
    'num_epochs': [100, 200],
    'batch_size': [16, 32]
}

# Exécuter l'optimisation
best_params, ensemble_pred, y_test = optimize_rnn(data, param_grid)

# Visualisation des prédictions ensemblées
plt.figure(figsize=(12, 6))
plt.plot(y_test, label='Valeurs réelles (Test)')
plt.plot(ensemble_pred, label='Prédictions Ensemblées (Test)')
plt.title('Prédictions Ensemblées vs Valeurs Réelles - Bitcoin Returns (BTC/EUR) with RNN')
plt.xlabel('Temps')
plt.ylabel('Rendement')
plt.legend()
plt.savefig('bitcoin_eur_returns_rnn_ensemble_day29.png')
plt.show()
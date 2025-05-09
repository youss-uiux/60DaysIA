import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import requests
import time
import random
import tensorflow.keras.backend as K
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from deap import base, creator, tools, algorithms
import os

# Fixer les seeds pour reproductibilité
random.seed(42)
np.random.seed(42)
os.environ['PYTHONHASHSEED'] = '42'

# 1. Chargement des données via API Alpha Vantage
api_key = 'YOUR_API_KEY'
url = f'https://www.alphavantage.co/query?function=DIGITAL_CURRENCY_DAILY&symbol=BTC&market=EUR&apikey={api_key}'

for _ in range(3):  # Tentatives multiples en cas d’échec réseau
    try:
        r = requests.get(url)
        data = r.json()
        if 'Time Series (Digital Currency Daily)' not in data:
            raise ValueError("Réponse invalide.")
        break
    except Exception as e:
        print(f"Erreur API, tentative suivante... ({e})")
        time.sleep(5)
else:
    raise RuntimeError("Impossible de récupérer les données après 3 tentatives.")

df = pd.DataFrame.from_dict(data['Time Series (Digital Currency Daily)'], orient='index').astype(float)
df.sort_index(inplace=True)

# 2. Extraction des rendements
close_column = next((col for col in ['4b. close (EUR)', '4a. close (EUR)', '4. close'] if col in df.columns), None)
if not close_column:
    raise KeyError("Colonne de clôture non trouvée.")
returns = np.diff(df[close_column].values)

# 3. Fonction de création des séquences
def create_sequences(data, num_lags):
    X, y = [], []
    for i in range(len(data) - num_lags):
        X.append(data[i:i + num_lags])
        y.append(data[i + num_lags])
    return np.array(X), np.array(y)

# 4. Préparation des données
train_test_split = 0.80
max_lags = 100
X_full, y_full = create_sequences(returns, max_lags)
train_size = int(len(X_full) * train_test_split)
X_train_full, X_test_full = X_full[:train_size], X_full[train_size:]
y_train_full, y_test_full = y_full[:train_size], y_full[train_size:]

# Normalisation des données
scaler_X = StandardScaler()
X_train_full = scaler_X.fit_transform(X_train_full)
X_test_full = scaler_X.transform(X_test_full)

scaler_y = StandardScaler()
y_train_full_scaled = scaler_y.fit_transform(y_train_full.reshape(-1, 1)).ravel()
y_test_full_scaled = scaler_y.transform(y_test_full.reshape(-1, 1)).ravel()

# 5. Configuration de l'algorithme génétique
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register("attr_lags", random.randint, 20, 100)
toolbox.register("attr_units", random.randint, 20, 100)
toolbox.register("attr_epochs", random.randint, 50, 200)
toolbox.register("attr_batch", random.choice, [16, 32, 64])
toolbox.register("individual", tools.initCycle, creator.Individual,
                 (toolbox.attr_lags, toolbox.attr_units, toolbox.attr_epochs, toolbox.attr_batch), n=1)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutUniformInt, low=[20, 20, 50, 16], up=[100, 100, 200, 64], indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=3)

# Fonction d’évaluation
def evaluate_individual(individual):
    K.clear_session()
    num_lags, num_units, num_epochs, batch_size = individual
    lag_diff = max_lags - num_lags
    X_train = X_train_full[:, lag_diff:lag_diff + num_lags]
    X_test = X_test_full[:, lag_diff:lag_diff + num_lags]

    X_train = X_train.reshape((X_train.shape[0], num_lags, 1))
    X_test = X_test.reshape((X_test.shape[0], num_lags, 1))

    model = Sequential()
    model.add(SimpleRNN(num_units, input_shape=(num_lags, 1)))
    model.add(Dropout(0.3))
    model.add(Dense(20, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')

    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    model.fit(X_train, y_train_full_scaled, epochs=num_epochs, batch_size=batch_size,
              validation_split=0.2, verbose=0, callbacks=[early_stopping])

    y_pred = model.predict(X_test, verbose=0).ravel()
    mse = mean_squared_error(y_test_full_scaled, y_pred)
    return mse,

toolbox.register("evaluate", evaluate_individual)

# 6. Algorithme génétique
population = toolbox.population(n=10)
hof = tools.HallOfFame(1)
stats = tools.Statistics(lambda ind: ind.fitness.values[0])
stats.register("avg", np.mean)
stats.register("min", np.min)

population, logbook = algorithms.eaSimple(population, toolbox, cxpb=0.5, mutpb=0.2,
                                           ngen=5, stats=stats, halloffame=hof, verbose=True)

# 7. Résultats finaux
best_ind = hof[0]
num_lags, num_units, num_epochs, batch_size = best_ind
print(f"Meilleur individu: lags={num_lags}, units={num_units}, epochs={num_epochs}, batch={batch_size}")

# Entraînement final avec les meilleurs paramètres
lag_diff = max_lags - num_lags
X_train = X_train_full[:, lag_diff:lag_diff + num_lags].reshape((-1, num_lags, 1))
X_test = X_test_full[:, lag_diff:lag_diff + num_lags].reshape((-1, num_lags, 1))

final_model = Sequential()
final_model.add(SimpleRNN(num_units, input_shape=(num_lags, 1)))
final_model.add(Dropout(0.3))
final_model.add(Dense(20, activation='relu'))
final_model.add(Dropout(0.3))
final_model.add(Dense(1))
final_model.compile(loss='mse', optimizer='adam')
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
final_model.fit(X_train, y_train_full_scaled, epochs=num_epochs, batch_size=batch_size,
                validation_split=0.2, verbose=0, callbacks=[early_stopping])

y_pred_final = final_model.predict(X_test).ravel()
y_pred_final_rescaled = scaler_y.inverse_transform(y_pred_final.reshape(-1, 1)).ravel()

# Sauvegarder le modèle
final_model.save("best_rnn_model_btc_eur.h5")

# Évaluer les performances
final_mse = mean_squared_error(y_test_full, y_pred_final_rescaled)
print(f"Meilleur MSE (non-scalé) : {final_mse:.6f}")

# 8. Visualisation
plt.figure(figsize=(12, 6))
plt.plot(y_test_full, label='Rendement réel')
plt.plot(y_pred_final_rescaled, label='Prédiction RNN optimisée')
plt.title('Prédiction du rendement Bitcoin (BTC/EUR)')
plt.xlabel('Temps')
plt.ylabel('Rendement')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("btc_eur_prediction_rnn_final.png")
plt.show()

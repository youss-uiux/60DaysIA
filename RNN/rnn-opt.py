import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import pandas as pd
import requests
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from deap import base, creator, tools, algorithms
import random

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

# Préparer les données une seule fois (pour éviter de recalculer à chaque individu)
train_test_split = 0.80
max_lags = 100  # Maximum pour l'optimisation
X, y = create_sequences(data, max_lags, train_test_split)
train_size = int(len(X) * train_test_split)
x_train_full, x_test_full = X[:train_size], X[train_size:]
y_train_full, y_test_full = y[:train_size], y[train_size:]

# Normalisation
scaler = StandardScaler()
x_train_full = scaler.fit_transform(x_train_full)
x_test_full = scaler.transform(x_test_full)

# Fonction de fitness (MSE Test)
def evaluate_individual(individual):
    num_lags, num_units, num_epochs, batch_size = individual
    
    # Ajuster num_lags pour les données
    lag_diff = max_lags - num_lags
    x_train = x_train_full[:, lag_diff:lag_diff+num_lags]
    x_test = x_test_full[:, lag_diff:lag_diff+num_lags]
    y_train = y_train_full
    y_test = y_test_full

    # Ajuster les dimensions pour le RNN
    x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
    x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))

    # Modèle RNN
    model = Sequential()
    model.add(SimpleRNN(num_units, input_shape=(num_lags, 1), return_sequences=False))
    model.add(Dropout(0.3))
    model.add(Dense(20, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')

    # Early stopping
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    # Entraîner le modèle
    history = model.fit(x_train, y_train.reshape(-1, 1), epochs=num_epochs, batch_size=batch_size,
                        validation_split=0.2, verbose=0, callbacks=[early_stopping])

    # Prédictions
    y_predicted = model.predict(x_test, verbose=0)
    mse_test = mean_squared_error(y_test, y_predicted)
    return mse_test,

# Configuration de l'algorithme génétique avec DEAP
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))  # Minimiser le MSE
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register("attr_lags", random.randint, 20, 100)  # num_lags entre 20 et 100
toolbox.register("attr_units", random.randint, 20, 100)  # num_units entre 20 et 100
toolbox.register("attr_epochs", random.randint, 50, 200)  # num_epochs entre 50 et 200
toolbox.register("attr_batch", random.choice, [16, 32, 64])  # batch_size parmi [16, 32, 64]
toolbox.register("individual", tools.initCycle, creator.Individual,
                 (toolbox.attr_lags, toolbox.attr_units, toolbox.attr_epochs, toolbox.attr_batch), n=1)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("evaluate", evaluate_individual)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutUniformInt, low=[20, 20, 50, 16], up=[100, 100, 200, 64], indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=3)

# Exécuter l'algorithme génétique
population = toolbox.population(n=10)  # Population initiale de 10 individus
ngen = 5  # Nombre de générations
all_predictions = []

for gen in range(ngen):
    offspring = algorithms.varAnd(population, toolbox, cxpb=0.5, mutpb=0.2)
    fits = toolbox.map(toolbox.evaluate, offspring)
    for fit, ind in zip(fits, offspring):
        ind.fitness.values = fit
        # Entraîner un modèle avec cet individu et stocker ses prédictions
        num_lags, num_units, num_epochs, batch_size = ind
        lag_diff = max_lags - num_lags
        x_train = x_train_full[:, lag_diff:lag_diff+num_lags]
        x_test = x_test_full[:, lag_diff:lag_diff+num_lags]
        x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
        x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))
        model = Sequential()
        model.add(SimpleRNN(num_units, input_shape=(num_lags, 1), return_sequences=False))
        model.add(Dropout(0.3))
        model.add(Dense(20, activation='relu'))
        model.add(Dropout(0.3))
        model.add(Dense(1))
        model.compile(loss='mean_squared_error', optimizer='adam')
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        model.fit(x_train, y_train_full.reshape(-1, 1), epochs=num_epochs, batch_size=batch_size,
                  validation_split=0.2, verbose=0, callbacks=[early_stopping])
        y_pred = model.predict(x_test, verbose=0)
        all_predictions.append(y_pred)
    population = toolbox.select(offspring, k=len(population))

# Sélectionner le meilleur individu
best_individual = tools.selBest(population, k=1)[0]
best_params = {'num_lags': best_individual[0], 'num_units': best_individual[1],
               'num_epochs': best_individual[2], 'batch_size': best_individual[3]}
best_mse = best_individual.fitness.values[0]

# Ensembling
ensemble_pred = np.mean(all_predictions, axis=0)
ensemble_mse = mean_squared_error(y_test_full, ensemble_pred)

print(f"Meilleur MSE Test: {best_mse:.6f} avec {best_params}")
print(f"MSE Test Ensembling: {ensemble_mse:.6f}")

# Visualisation
plt.figure(figsize=(12, 6))
plt.plot(y_test_full, label='Valeurs réelles (Test)')
plt.plot(ensemble_pred, label='Prédictions Ensemblées (Test)')
plt.title('Prédictions Ensemblées vs Valeurs Réelles - Bitcoin Returns (BTC/EUR) with RNN')
plt.xlabel('Temps')
plt.ylabel('Rendement')
plt.legend()
plt.savefig('bitcoin_eur_returns_rnn_ensemble_day29.png')
plt.show()
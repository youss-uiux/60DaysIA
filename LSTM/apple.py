import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from sklearn.metrics import mean_squared_error

# 1. Récupérer les données
print("Étape 1 : Récupération des données")
start_date = '2015-01-01'
end_date = '2023-12-31'
data = yf.download('AAPL', start=start_date, end=end_date)
series = data['Close']

# Vérification des données
print(f"Longueur de la série : {len(series)}")
print(f"Valeurs manquantes : {series.isna().sum()}")
print(f"Échantillon des 5 premières valeurs :\n{series.head()}")

# 2. Préparer les données
def prepare_data(series, num_lags, train_test_split):
    data = series.values
    X, y = [], []
    for i in range(len(data) - num_lags):
        X.append(data[i:i + num_lags])
        y.append(data[i + num_lags])
    X = np.array(X)
    y = np.array(y)
    train_size = int(len(X) * train_test_split)
    x_train, x_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    x_train = x_train.reshape(-1, num_lags, 1)
    x_test = x_test.reshape(-1, num_lags, 1)
    return x_train, x_test, y_train, y_test

num_lags = 100
train_test_split = 0.80
x_train, x_test, y_train, y_test = prepare_data(series, num_lags, train_test_split)

# Vérification de la préparation des données
print("\nÉtape 2 : Vérification de la préparation des données")
print(f"Forme de x_train : {x_train.shape}")
print(f"Forme de x_test : {x_test.shape}")
print(f"Forme de y_train : {y_train.shape}")
print(f"Forme de y_test : {y_test.shape}")
print(f"Valeurs manquantes dans x_train : {np.isnan(x_train).sum()}")
print(f"Valeurs manquantes dans y_train : {np.isnan(y_train).sum()}")
print(f"Valeurs manquantes dans x_test : {np.isnan(x_test).sum()}")
print(f"Valeurs manquantes dans y_test : {np.isnan(y_test).sum()}")

# 3. Construire et entraîner le modèle LSTM
print("\nÉtape 3 : Construction et entraînement du modèle")
num_neurons_in_hidden_layers = 20
num_epochs = 100
batch_size = 32

model = Sequential()
model.add(LSTM(units=num_neurons_in_hidden_layers, input_shape=(num_lags, 1)))
model.add(Dense(num_neurons_in_hidden_layers, activation='relu'))
model.add(Dense(units=1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(x_train, y_train, epochs=num_epochs, batch_size=batch_size, verbose=1)

# Vérification du modèle
print("\nRésumé du modèle :")
model.summary()

# 4. Faire des prédictions
print("\nÉtape 4 : Génération des prédictions")
y_predicted_train = model.predict(x_train)
y_predicted = model.predict(x_test)
y_predicted_train = y_predicted_train.reshape(-1, 1)
y_predicted = y_predicted.reshape(-1, 1)

# Vérification des prédictions
print(f"Forme de y_predicted_train : {y_predicted_train.shape}")
print(f"Forme de y_predicted : {y_predicted.shape}")
print(f"Valeurs manquantes dans y_predicted_train : {np.isnan(y_predicted_train).sum()}")
print(f"Valeurs manquantes dans y_predicted : {np.isnan(y_predicted).sum()}")
print(f"Échantillon des 5 premières prédictions (train) :\n{y_predicted_train[:5].flatten()}")
print(f"Échantillon des 5 premières vraies valeurs (train) :\n{y_train[:5]}")
print(f"Échantillon des 5 premières prédictions (test) :\n{y_predicted[:5].flatten()}")
print(f"Échantillon des 5 premières vraies valeurs (test) :\n{y_test[:5]}")

# 5. Évaluer les performances
print("\nÉtape 5 : Calcul des métriques")
def calculate_accuracy(y_true, y_pred):
    threshold = 0.1 * np.mean(y_true)
    correct = np.abs(y_true - y_pred) < threshold
    return np.mean(correct) * 100

def model_bias(y_true, y_pred):
    return np.mean(y_pred - y_true)

# Recalculer chaque métrique
accuracy_train = calculate_accuracy(y_train, y_predicted_train)
accuracy_test = calculate_accuracy(y_test, y_predicted)
rmse_train = np.sqrt(mean_squared_error(y_train, y_predicted_train))
rmse_test = np.sqrt(mean_squared_error(y_test, y_predicted))
corr_train = np.corrcoef(y_train.flatten(), y_predicted_train.flatten())[0, 1]
corr_test = np.corrcoef(y_test.flatten(), y_predicted.flatten())[0, 1]
bias = model_bias(y_test, y_predicted)

# Afficher les métriques
print(f"Accuracy Train = {accuracy_train:.2f} %")
print(f"Accuracy Test = {accuracy_test:.2f} %")
print(f"RMSE Train = {rmse_train:.2f}")
print(f"RMSE Test = {rmse_test:.2f}")
print(f"Correlation In-Sample Predicted/Train = {corr_train:.3f}")
print(f"Correlation Out-of-Sample Predicted/Test = {corr_test:.3f}")
print(f"Model Bias = {bias:.2f}")

# Vérification manuelle des métriques
print("\nVérification manuelle des métriques")
# Précision
threshold_train = 0.1 * np.mean(y_train)
errors_train = np.abs(y_train - y_predicted_train.flatten())
correct_train = errors_train < threshold_train
manual_accuracy_train = np.mean(correct_train) * 100
print(f"Précision manuelle (train) = {manual_accuracy_train:.2f} %")

threshold_test = 0.1 * np.mean(y_test)
errors_test = np.abs(y_test - y_predicted.flatten())
correct_test = errors_test < threshold_test
manual_accuracy_test = np.mean(correct_test) * 100
print(f"Précision manuelle (test) = {manual_accuracy_test:.2f} %")

# RMSE
manual_rmse_train = np.sqrt(np.mean((y_train - y_predicted_train.flatten())**2))
manual_rmse_test = np.sqrt(np.mean((y_test - y_predicted.flatten())**2))
print(f"RMSE manuel (train) = {manual_rmse_train:.2f}")
print(f"RMSE manuel (test) = {manual_rmse_test:.2f}")

# Corrélation
manual_corr_train = np.corrcoef(y_train.flatten(), y_predicted_train.flatten())[0, 1]
manual_corr_test = np.corrcoef(y_test.flatten(), y_predicted.flatten())[0, 1]
print(f"Corrélation manuelle (train) = {manual_corr_train:.3f}")
print(f"Corrélation manuelle (test) = {manual_corr_test:.3f}")

# Biais
manual_bias = np.mean(y_predicted.flatten() - y_test)
print(f"Biais manuel = {manual_bias:.2f}")

# 6. Visualiser les résultats
print("\nÉtape 6 : Visualisation")
def plot_train_test_values(y_train, y_test, y_predicted_train, y_predicted):
    plt.figure(figsize=(12, 6))
    plt.plot(range(len(y_train)), y_train, label='Train True', color='blue')
    plt.plot(range(len(y_predicted_train)), y_predicted_train, label='Train Predicted', color='cyan', linestyle='--')
    test_start = len(y_train)
    plt.plot(range(test_start, test_start + len(y_test)), y_test, label='Test True', color='green')
    plt.plot(range(test_start, test_start + len(y_predicted)), y_predicted, label='Test Predicted', color='red', linestyle='--')
    plt.title('LSTM Time Series Forecasting')
    plt.xlabel('Time Steps')
    plt.ylabel('Value')
    plt.legend()
    plt.show()

plot_train_test_values(y_train, y_test, y_predicted_train, y_predicted)

# 7. Vérification supplémentaire : échantillon des erreurs
print("\nÉtape 7 : Analyse des erreurs")
errors_train = y_train - y_predicted_train.flatten()
errors_test = y_test - y_predicted.flatten()
print(f"Erreur moyenne (train) : {np.mean(errors_train):.2f}")
print(f"Erreur max (train) : {np.max(np.abs(errors_train)):.2f}")
print(f"Erreur moyenne (test) : {np.mean(errors_test):.2f}")
print(f"Erreur max (test) : {np.max(np.abs(errors_test)):.2f}")
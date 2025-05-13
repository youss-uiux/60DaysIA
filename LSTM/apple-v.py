import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression

# Fixer les graines pour la reproductibilité
np.random.seed(42)
import tensorflow as tf
tf.random.set_seed(42)

# Fonctions utilitaires
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

def calculate_accuracy(y_true, y_pred):
    threshold = 0.1 * np.mean(y_true)
    correct = np.abs(y_true - y_pred) < threshold
    return np.mean(correct) * 100

def model_bias(y_true, y_pred):
    return np.mean(y_pred - y_true)

# 1. Récupérer les données
print("Étape 1 : Récupération des données")
start_date = '2015-01-01'
end_date = '2023-12-31'
data = yf.download('AAPL', start=start_date, end=end_date)
series = data['Close']

# Normalisation
scaler = MinMaxScaler()
series_scaled = scaler.fit_transform(series.values.reshape(-1, 1)).flatten()
series = pd.Series(series_scaled, index=series.index)

# Préparer les données
num_lags = 100
train_test_split = 0.80
x_train, x_test, y_train, y_test = prepare_data(series, num_lags, train_test_split)

# 2. Entraîner le modèle LSTM
print("\nÉtape 2 : Entraînement du modèle LSTM")
num_neurons_in_hidden_layers = 20
num_epochs = 100
batch_size = 32

model = Sequential()
model.add(LSTM(units=num_neurons_in_hidden_layers, input_shape=(num_lags, 1)))
model.add(Dense(num_neurons_in_hidden_layers, activation='relu'))
model.add(Dense(units=1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(x_train, y_train, epochs=num_epochs, batch_size=batch_size, verbose=1)

# Prédictions
y_predicted_train = model.predict(x_train)
y_predicted = model.predict(x_test)
y_predicted_train = scaler.inverse_transform(y_predicted_train).flatten()
y_predicted = scaler.inverse_transform(y_predicted).flatten()
y_train_unscaled = scaler.inverse_transform(y_train.reshape(-1, 1)).flatten()
y_test_unscaled = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()

# Métriques LSTM
accuracy_train = calculate_accuracy(y_train_unscaled, y_predicted_train)
accuracy_test = calculate_accuracy(y_test_unscaled, y_predicted)
rmse_train = np.sqrt(mean_squared_error(y_train_unscaled, y_predicted_train))
rmse_test = np.sqrt(mean_squared_error(y_test_unscaled, y_predicted))
corr_train = np.corrcoef(y_train_unscaled, y_predicted_train)[0, 1]
corr_test = np.corrcoef(y_test_unscaled, y_predicted)[0, 1]
bias = model_bias(y_test_unscaled, y_predicted)

print("\nMétriques LSTM :")
print(f"Accuracy Train = {accuracy_train:.2f} %")
print(f"Accuracy Test = {accuracy_test:.2f} %")
print(f"RMSE Train = {rmse_train:.2f}")
print(f"RMSE Test = {rmse_test:.2f}")
print(f"Correlation Train = {corr_train:.3f}")
print(f"Correlation Test = {corr_test:.3f}")
print(f"Model Bias = {bias:.2f}")

# 3. Validation croisée temporelle
print("\nÉtape 3 : Validation croisée temporelle")
n_folds = 5
fold_size = len(series) // n_folds
rmse_folds = []

for i in range(n_folds):
    start_idx = i * fold_size
    end_idx = (i + 1) * fold_size if i < n_folds - 1 else len(series)
    train_series = pd.concat([series[:start_idx], series[end_idx:]])
    test_series = series[start_idx:end_idx]
    
    x_train_fold, x_test_fold, y_train_fold, y_test_fold = prepare_data(test_series, num_lags, 0.8)
    
    model_fold = Sequential()
    model_fold.add(LSTM(units=num_neurons_in_hidden_layers, input_shape=(num_lags, 1)))
    model_fold.add(Dense(num_neurons_in_hidden_layers, activation='relu'))
    model_fold.add(Dense(units=1))
    model_fold.compile(loss='mean_squared_error', optimizer='adam')
    model_fold.fit(x_train_fold, y_train_fold, epochs=num_epochs, batch_size=batch_size, verbose=0)
    
    y_pred_fold = model_fold.predict(x_test_fold)
    y_pred_fold = scaler.inverse_transform(y_pred_fold).flatten()
    y_test_fold_unscaled = scaler.inverse_transform(y_test_fold.reshape(-1, 1)).flatten()
    
    rmse_fold = np.sqrt(mean_squared_error(y_test_fold_unscaled, y_pred_fold))
    rmse_folds.append(rmse_fold)
    print(f"Fold {i+1} RMSE = {rmse_fold:.2f}")

print(f"Moyenne RMSE (validation croisée) = {np.mean(rmse_folds):.2f}")

# 4. Comparaison avec régression linéaire
print("\nÉtape 4 : Comparaison avec régression linéaire")
model_lr = LinearRegression()
model_lr.fit(x_train.reshape(x_train.shape[0], -1), y_train)
y_pred_lr_train = model_lr.predict(x_train.reshape(x_train.shape[0], -1))
y_pred_lr_test = model_lr.predict(x_test.reshape(x_test.shape[0], -1))

y_pred_lr_train = scaler.inverse_transform(y_pred_lr_train.reshape(-1, 1)).flatten()
y_pred_lr_test = scaler.inverse_transform(y_pred_lr_test.reshape(-1, 1)).flatten()

rmse_lr_train = np.sqrt(mean_squared_error(y_train_unscaled, y_pred_lr_train))
rmse_lr_test = np.sqrt(mean_squared_error(y_test_unscaled, y_pred_lr_test))
print(f"Régression linéaire RMSE Train = {rmse_lr_train:.2f}")
print(f"Régression linéaire RMSE Test = {rmse_lr_test:.2f}")

# 5. Test sur une nouvelle période
print("\nÉtape 5 : Test sur 2024")
start_date_new = '2024-01-01'
end_date_new = '2024-12-31'
data_new = yf.download('AAPL', start=start_date_new, end=end_date_new)
series_new = data_new['Close']
series_new_scaled = scaler.transform(series_new.values.reshape(-1, 1)).flatten()
series_new = pd.Series(series_new_scaled, index=series_new.index)

x_new, _, y_new, _ = prepare_data(series_new, num_lags, 1.0)  # Pas de test
y_pred_new = model.predict(x_new)
y_pred_new = scaler.inverse_transform(y_pred_new).flatten()
y_new_unscaled = scaler.inverse_transform(y_new.reshape(-1, 1)).flatten()

rmse_new = np.sqrt(mean_squared_error(y_new_unscaled, y_pred_new))
print(f"RMSE sur 2024 = {rmse_new:.2f}")

# Visualisation
plt.figure(figsize=(12, 6))
plt.plot(range(len(y_new_unscaled)), y_new_unscaled, label='True 2024', color='green')
plt.plot(range(len(y_pred_new)), y_pred_new, label='Predicted 2024', color='red', linestyle='--')
plt.title('LSTM Forecasting on 2024 Data')
plt.xlabel('Time Steps')
plt.ylabel('Value (USD)')
plt.legend()
plt.show()
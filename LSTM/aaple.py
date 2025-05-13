import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from sklearn.metrics import mean_squared_error

# 1. Récupérer les données
start_date = '2015-01-01'
end_date = '2023-12-31'
data = yf.download('AAPL', start=start_date, end=end_date)
series = data['Close']

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

# 3. Construire et entraîner le modèle LSTM
num_neurons_in_hidden_layers = 20
num_epochs = 100
batch_size = 32

model = Sequential()
model.add(LSTM(units=num_neurons_in_hidden_layers, input_shape=(num_lags, 1)))
model.add(Dense(num_neurons_in_hidden_layers, activation='relu'))
model.add(Dense(units=1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(x_train, y_train, epochs=num_epochs, batch_size=batch_size, verbose=1)

# 4. Faire des prédictions
y_predicted_train = model.predict(x_train)
y_predicted = model.predict(x_test)
y_predicted_train = y_predicted_train.reshape(-1, 1)
y_predicted = y_predicted.reshape(-1, 1)

# 5. Évaluer les performances
def calculate_accuracy(y_true, y_pred):
    threshold = 0.1 * np.mean(y_true)
    correct = np.abs(y_true - y_pred) < threshold
    return np.mean(correct) * 100

def model_bias(y_true, y_pred):
    return np.mean(y_pred - y_true)

accuracy_train = calculate_accuracy(y_train, y_predicted_train)
accuracy_test = calculate_accuracy(y_test, y_predicted)
rmse_train = np.sqrt(mean_squared_error(y_train, y_predicted_train))
rmse_test = np.sqrt(mean_squared_error(y_test, y_predicted))
corr_train = np.corrcoef(y_train.flatten(), y_predicted_train.flatten())[0, 1]
corr_test = np.corrcoef(y_test.flatten(), y_predicted.flatten())[0, 1]
bias = model_bias(y_test, y_predicted)

print(f"Accuracy Train = {accuracy_train:.2f} %")
print(f"Accuracy Test = {accuracy_test:.2f} %")
print(f"RMSE Train = {rmse_train:.2f}")
print(f"RMSE Test = {rmse_test:.2f}")
print(f"Correlation In-Sample Predicted/Train = {corr_train:.3f}")
print(f"Correlation Out-of-Sample Predicted/Test = {corr_test:.3f}")
print(f"Model Bias = {bias:.2f}")

# 6. Visualiser les résultats
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
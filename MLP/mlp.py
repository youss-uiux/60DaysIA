# ------------------------- 0. Imports -------------------------
import time
import requests_cache
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
import tensorflow as tf

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import recall_score, precision_score, roc_auc_score, accuracy_score, confusion_matrix, roc_curve
from imblearn.over_sampling import SMOTE
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2

# ------------------------- 1. Cache HTTP -------------------------
requests_cache.install_cache("yfinance_cache", expire_after=86400)  # 24 h

# ------------------------- 2. Téléchargement avec retry -------------------------
def download_sp500(ticker="^GSPC", start="2015-01-01", end="2025-01-01", n_try=5, sleep_base=2):
    for attempt in range(1, n_try + 1):
        try:
            # Désactive auto_adjust explicitement et récupère tous les prix
            df = yf.download(ticker, start=start, end=end, progress=False, 
                            threads=False, auto_adjust=False, actions=False)
            if not df.empty:
                print(f"Téléchargement réussi à la tentative {attempt}")
                return df[['Open', 'High', 'Low', 'Close', 'Volume']]  # On ne garde que les colonnes nécessaires
        except Exception as e:
            print(f"[Tentative {attempt}/{n_try}] Erreur: {e}")
            if "Rate limited" in str(e):
                wait_time = sleep_base ** attempt
                print(f"Rate limit atteint, attente de {wait_time} secondes...")
                time.sleep(wait_time)
    raise RuntimeError("Impossible de télécharger les données après plusieurs tentatives.")

# ------------------------- 3. Chargement des données -------------------------
try:
    sp500 = download_sp500()
except RuntimeError as e:
    print(f"⚠️ Erreur de téléchargement: {e}")
    print("Utilisation d'une sauvegarde locale sp500_backup.csv")
    try:
        sp500 = pd.read_csv("sp500_backup.csv", parse_dates=["Date"], index_col="Date")
    except FileNotFoundError:
        raise RuntimeError("Aucune sauvegarde locale trouvée. Impossible de continuer.")

# Création de features supplémentaires
data = sp500.copy()

# Features techniques
data['SMA_5'] = data['Close'].rolling(window=5).mean()
data['SMA_20'] = data['Close'].rolling(window=20).mean()
data['RSI'] = 100 - 100 / (1 + (data['Close'].diff(1).clip(lower=0).rolling(window=14).mean() /  data['Close'].diff(1).clip(upper=0).abs().rolling(window=14).mean()))
data['MACD'] = data['Close'].ewm(span=12, adjust=False).mean() - data['Close'].ewm(span=26, adjust=False).mean()

# Correction des bandes de Bollinger
rolling_std = data['Close'].rolling(window=20).std()
data['Bollinger_Upper'] = data['SMA_20'] + (2 * rolling_std)
data['Bollinger_Lower'] = data['SMA_20'] - (2 * rolling_std)

# Target: 1 si le cours monte le jour suivant, 0 sinon
data["Target"] = np.where(data["Close"].shift(-1) > data["Close"], 1, 0)
data.dropna(inplace=True)

if data.empty:
    raise RuntimeError("Le DataFrame est vide - vérifiez la source de données.")

# ------------------------- 4. Split chronologique -------------------------
features = ["Open", "High", "Low", "Close", "Volume", "SMA_5", "SMA_20", 
            "RSI", "MACD", "Bollinger_Upper", "Bollinger_Lower"]
split_date = "2023-01-01"  # Plus de données pour l'entraînement

train = data.loc[:split_date]
test = data.loc[split_date:]

X_train, y_train = train[features], train["Target"]
X_test, y_test = test[features], test["Target"]

# Vérification du déséquilibre de classes
print("\nDistribution des classes:")
print(f"Train - Hausse: {y_train.mean():.2%}, Baisse: {1-y_train.mean():.2%}")
print(f"Test - Hausse: {y_test.mean():.2%}, Baisse: {1-y_test.mean():.2%}")

# ------------------------- 5. Standardisation -------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Rééchantillonnage pour équilibrer les classes
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)

# ------------------------- 6. Modèle MLP amélioré -------------------------
model = Sequential([
    Dense(64, activation="relu", input_shape=(X_train_resampled.shape[1],), kernel_regularizer=l2(0.001)),
    BatchNormalization(),
    Dropout(0.5),
    
    Dense(32, activation="relu", kernel_regularizer=l2(0.001)),
    BatchNormalization(),
    Dropout(0.3),
    
    Dense(16, activation="relu", kernel_regularizer=l2(0.001)),
    BatchNormalization(),
    
    Dense(1, activation="sigmoid")
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="binary_crossentropy",
    metrics=["accuracy", tf.keras.metrics.AUC(name='auc')]
)

early_stopping = EarlyStopping(
    monitor='val_auc',
    patience=10,
    mode='max',
    restore_best_weights=True
)

history = model.fit(
    X_train_resampled, y_train_resampled,
    validation_split=0.2,
    epochs=100,
    batch_size=64,
    verbose=1,
    callbacks=[early_stopping]
)

# ------------------------- 7. Évaluation -------------------------
y_pred_proba = model.predict(X_test_scaled).flatten()
y_pred = (y_pred_proba >= 0.5).astype(int)

recall = recall_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred_proba)
accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print("\nMétriques sur l'ensemble de test :")
print(f"  Rappel     : {recall:.2f}")
print(f"  Précision  : {precision:.2f}")
print(f"  AUC        : {auc:.2f}")
print(f"  Exactitude : {accuracy:.2f}")
print("Matrice de confusion :")
print(cm)

# ------------------------- 8. Visualisations -------------------------
# Matrice de confusion
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Baisse", "Hausse"],
            yticklabels=["Baisse", "Hausse"])
plt.title("Matrice de confusion - Prédiction S&P 500 (MLP amélioré)")
plt.xlabel("Prédictions")
plt.ylabel("Réel")
plt.tight_layout()
plt.savefig("confusion_matrix_sp500_improved.png")
plt.show()

# Courbe ROC
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, lw=2, label=f"AUC = {auc:.2f}")
plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("Taux de faux positifs (FPR)")
plt.ylabel("Taux de vrais positifs (TPR)")
plt.title("Courbe ROC - Prédiction S&P 500 (MLP amélioré)")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig("roc_curve_sp500_improved.png")
plt.show()
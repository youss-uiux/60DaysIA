# ------------------------- 0. Imports -------------------------
import time
import requests_cache
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import recall_score, precision_score, roc_auc_score, accuracy_score, confusion_matrix, roc_curve
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

# ------------------------- 1. Cache HTTP -------------------------
requests_cache.install_cache("yfinance_cache", expire_after=86400)  # 24 h

# ------------------------- 2. Téléchargement avec retry -------------------------
def download_sp500(ticker="^GSPC", start="2020-01-01", end="2025-01-01", n_try=5, sleep_base=2):
    for attempt in range(1, n_try + 1):
        try:
            df = yf.download(ticker, start=start, end=end, progress=False, threads=False)
            if not df.empty:
                print(f"Téléchargement réussi à la tentative {attempt}")
                return df
        except Exception as e:
            print(f"[Tentative {attempt}/{n_try}] Erreur: {e}")
        time.sleep(sleep_base ** attempt)  # backoff exponentiel
    raise RuntimeError("Impossible de télécharger les données après plusieurs tentatives.")

# ------------------------- 3. Chargement des données -------------------------
try:
    sp500 = download_sp500()
except RuntimeError:
    print("⚠️ Utilisation d'une sauvegarde locale sp500_backup.csv")
    sp500 = pd.read_csv("sp500_backup.csv", parse_dates=["Date"], index_col="Date")

data = sp500[["Open", "High", "Low", "Close", "Volume"]].copy()
data["Target"] = np.where(data["Close"].shift(-1) > data["Close"], 1, 0)
data.dropna(inplace=True)

if data.empty:
    raise RuntimeError("Le DataFrame est vide – vérifie la source de données.")

# ------------------------- 4. Split chronologique -------------------------
features = ["Open", "High", "Low", "Close", "Volume"]
split_date = "2024-01-01"

train = data.loc[:split_date]
test = data.loc[split_date:]

X_train, y_train = train[features], train["Target"]
X_test, y_test = test[features], test["Target"]

# ------------------------- 5. Standardisation -------------------------
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ------------------------- 6. Modèle MLP -------------------------
model = Sequential([
    Dense(32, activation="relu", input_shape=(X_train.shape[1],)),
    Dropout(0.3),
    Dense(16, activation="relu"),
    Dropout(0.2),
    Dense(8, activation="relu"),
    Dense(1, activation="sigmoid")
])

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

history = model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=20,
    batch_size=32,
    verbose=1
)

# ------------------------- 7. Évaluation -------------------------
y_pred_proba = model.predict(X_test).flatten()
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
plt.title("Matrice de confusion – Prédiction S&P 500 (MLP)")
plt.xlabel("Prédictions")
plt.ylabel("Réel")
plt.tight_layout()
plt.savefig("confusion_matrix_sp500.png")
plt.show()

# Courbe ROC
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, lw=2, label=f"AUC = {auc:.2f}")
plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("Taux de faux positifs (FPR)")
plt.ylabel("Taux de vrais positifs (TPR)")
plt.title("Courbe ROC – Prédiction S&P 500 (MLP)")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig("roc_curve_sp500.png")
plt.show()

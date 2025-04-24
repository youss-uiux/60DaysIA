import yfinance as yf
from hmmlearn import hmm
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

data = yf.download("AAPL", start="2022-01-01", end="2023-12-31")

data['Variation'] = data['Close'].diff()

seuil = data["Variation"].std() * 0.5
def classify_change(change, seuil=0.5):
    if pd.isna(change):
        return None
    elif change > seuil:
        return "↑"
    elif change < -seuil:
        return "↓"
    else:
        return "="


data['Observation'] = data['Variation'].apply(lambda x: classify_change(x, seuil))

result = data[['Close', 'Variation', 'Observation']].dropna()

# Vérification des observations
print(data["Observation"].value_counts())
print(result.head())

symboles = ["↓", "=", "↑"]
encoder = LabelEncoder()
encoded_obs = encoder.fit_transform(result["Observation"])

X = encoded_obs.reshape(-1, 1)

#n_states = 3 

#model = hmm.MultinomialHMM(n_components=n_states, n_iter=1000, random_state=42)

#model.fit(X)

model = hmm.MultinomialHMM(n_components=3, n_iter=1000)
model.startprob_ = np.array([0.33, 0.33, 0.34])  # Probabilités initiales équilibrées
model.transmat_ = np.array([[0.6, 0.2, 0.2], [0.2, 0.6, 0.2], [0.2, 0.2, 0.6]])  # Encourage les transitions
model.fit(X)

hidden_states = model.predict(X)

result["État caché (HMM)"] = hidden_states

print(result.head(10))


# Affecter une couleur différente à chaque état caché
colors = ["#ff9999", "#99ccff", "#aaffaa"]  # rouge = baissier, bleu = stable, vert = haussier

# Récupérer la séquence des états et des prix
prix = result["Close"].values
etats = result["État caché (HMM)"].values
dates = result.index

plt.figure(figsize=(14, 6))
for i in range(len(prix) - 1):
    plt.plot(dates[i:i+2], prix[i:i+2],
             color=colors[etats[i]], linewidth=2)
    

patches = [mpatches.Patch(color=c, label=f'État {i}') for i, c in enumerate(colors)]
plt.legend(handles=patches)

plt.title("État caché du marché prédit par le HMM (action Apple)")
plt.xlabel("Date")
plt.ylabel("Prix de clôture ($)")
plt.grid(True)
plt.tight_layout()
plt.show()



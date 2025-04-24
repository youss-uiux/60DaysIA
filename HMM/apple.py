import yfinance as yf
import pandas as pd

data = yf.download("AAPL", start="2022-01-01", end="2023-12-31")

data['Variation'] = data['Close'].diff()

def classify_change(change, seuil=0.5):
    if pd.isna(change):
        return None
    elif change > seuil:
        return "↑"
    elif change < -seuil:
        return "↓"
    else:
        return "="


data['Observation'] = data['Variation'].apply(classify_change)


result = data[['Close', 'Variation', 'Observation']].dropna()

print(result.head())
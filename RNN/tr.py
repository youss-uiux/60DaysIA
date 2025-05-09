import re
import pandas as pd

# === Charger la discussion (copie depuis fichier texte brut) ===
with open("_chat.txt", "r", encoding="utf-8") as f:
    lines = f.readlines()

# === Nettoyer et parser ===
conversations = []
pattern = re.compile(r"^\[\d{2}/\d{2}/\d{4} \d{2}:\d{2}:\d{2}] ([^:]+): (.+)$")

parsed = []
for line in lines:
    match = pattern.match(line.strip())
    if match:
        speaker = match.group(1)
        message = match.group(2).strip()
        parsed.append((speaker, message))

# === Extraire les paires question -> réponse ===
pairs = []
for i in range(len(parsed) - 1):
    sender1, msg1 = parsed[i]
    sender2, msg2 = parsed[i + 1]
    # Exemple : Youss pose une question, Ouaraqua répond
    if sender1.lower().startswith("youss") and sender2.lower().startswith("ouaraqua"):
        # Optionnel : ignorer les messages système, vocaux, etc.
        if msg1 and msg2 and len(msg1) < 300 and len(msg2) < 300:
            pairs.append((msg1, msg2))

# === Afficher / Sauvegarder les paires extraites ===
df = pd.DataFrame(pairs, columns=["Message de Youss", "Réponse de Ouaraqua"])
print(df.head(10))

# === Sauvegarder pour entraîner un modèle ===
df.to_csv("paires_youss_ouaraqua.csv", index=False)

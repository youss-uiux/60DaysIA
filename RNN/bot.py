import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense

# === 1. Charger le CSV avec les paires de dialogue ===
df = pd.read_csv("paires_youss_ouaraqua.csv")
inputs = df["Message de Youss"].astype(str).tolist()
outputs = df["Réponse de Ouaraqua"].astype(str).tolist()

# === 2. Prétraitement : tokenisation et encodage ===
tokenizer = Tokenizer()
tokenizer.fit_on_texts(inputs + outputs)
vocab_size = len(tokenizer.word_index) + 1

input_sequences = tokenizer.texts_to_sequences(inputs)
output_sequences = tokenizer.texts_to_sequences(outputs)

max_len_input = max(len(seq) for seq in input_sequences)
max_len_output = max(len(seq) for seq in output_sequences)

X = pad_sequences(input_sequences, maxlen=max_len_input, padding='post')
y = pad_sequences(output_sequences, maxlen=max_len_output, padding='post')

# === 3. RNN simple (sortie = 1er mot de la réponse) ===
# Pour simplicité, on entraîne à prédire juste le premier mot de la réponse
y_single = np.array([seq[0] for seq in y])  # on prend juste le premier mot

model = Sequential()
model.add(Embedding(vocab_size, 64, input_length=max_len_input))
model.add(SimpleRNN(128))
model.add(Dense(vocab_size, activation='softmax'))
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# === 4. Entraînement ===
model.fit(X, y_single, epochs=300, verbose=1)

# === 5. Génération de réponse ===
reverse_word_index = {i: w for w, i in tokenizer.word_index.items()}

def generate_reply(prompt):
    seq = tokenizer.texts_to_sequences([prompt.lower()])
    seq = pad_sequences(seq, maxlen=max_len_input, padding='post')
    pred = model.predict(seq, verbose=0)
    word_id = np.argmax(pred)
    return reverse_word_index.get(word_id, "🤷")

# === 6. Dialogue avec le bot ===
print("\n=== Chatbot Ouaraqua est prêt ===")
while True:
    user_input = input("Toi : ")
    if user_input.lower() in ['exit', 'quit']:
        break
    response = generate_reply(user_input)
    print("Ouaraqua :", response)

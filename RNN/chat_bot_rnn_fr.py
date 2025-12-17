import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader

# -----------------------------
# 1. Chargement du dataset français
# -----------------------------
print("Téléchargement du dataset Claire-Dialogue-French-0.1...")
dataset = load_dataset("OpenLLM-France/Claire-Dialogue-French-0.1", sample_by="paragraph", streaming=True)

# Extraction des paires consécutives (tour i → tour i+1)
pairs = []
max_pairs = 20000  # Limite pour éviter de saturer la RAM (augmentez si vous avez plus de RAM/GPU)
count = 0

for dialog in dataset:
    turns = dialog["turns"]  # chaque turn a une clé "utterance"
    utterances = [turn["utterance"].strip().lower() for turn in turns if turn["utterance"].strip()]
    for i in range(len(utterances) - 1):
        if len(utterances[i].split()) > 1 and len(utterances[i+1].split()) > 1:  # phrases pas trop courtes
            pairs.append([utterances[i], utterances[i+1]])
            count += 1
            if count >= max_pairs:
                break
    if count >= max_pairs:
        break

print(f"{len(pairs)} paires extraites pour l'entraînement.")

# -----------------------------
# 2. Construction du vocabulaire
# -----------------------------
SOS_token = 0
EOS_token = 1

class Lang:
    def __init__(self):
        self.word2index = {}
        self.word2count = {}
        self.index2word = {SOS_token: "SOS", EOS_token: "EOS"}
        self.n_words = 2  # SOS + EOS

    def add_sentence(self, sentence):
        for word in sentence.split(' '):
            self.add_word(word)

    def add_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

lang = Lang()
for pair in pairs:
    lang.add_sentence(pair[0])
    lang.add_sentence(pair[1])

print(f"Vocabulaire : {lang.n_words} mots uniques.")

# -----------------------------
# 3. Conversion phrase → tenseur
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def tensor_from_sentence(lang, sentence):
    indexes = [lang.word2index[word] for word in sentence.split(' ')]
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(1, -1)

# Préparer les paires d'entraînement
training_pairs = [ (tensor_from_sentence(lang, pair[0]), tensor_from_sentence(lang, pair[1]))
                  for pair in pairs ]

# -----------------------------
# 4. Modèles Encodeur et Décodeur
# -----------------------------
class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)

    def forward(self, input_tensor):
        embedded = self.embedding(input_tensor)
        _, hidden = self.gru(embedded)
        return hidden  # hidden final (1, 1, hidden_size)

class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(DecoderRNN, self).__init__()
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, input_tensor, hidden):
        embedded = self.embedding(input_tensor)
        output, hidden = self.gru(embedded, hidden)
        output = self.out(output.squeeze(1))  # (1, vocab_size)
        return output, hidden

# -----------------------------
# 5. Fonction d'entraînement
# -----------------------------
hidden_size = 256
max_length = 20  # Longueur max pour le décodage

encoder = EncoderRNN(lang.n_words, hidden_size).to(device)
decoder = DecoderRNN(hidden_size, lang.n_words).to(device)

encoder_optimizer = optim.SGD(encoder.parameters(), lr=0.01)
decoder_optimizer = optim.SGD(decoder.parameters(), lr=0.01)

def train_one_pair(input_tensor, target_tensor):
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    loss = 0
    criterion = nn.CrossEntropyLoss()

    # Encodeur
    encoder_hidden = encoder(input_tensor)

    # Décodeur (teacher forcing)
    decoder_input = torch.tensor([[SOS_token]], device=device)
    decoder_hidden = encoder_hidden

    target_length = min(target_tensor.size(1), max_length)

    for di in range(target_length):
        decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
        topv, topi = decoder_output.topk(1)
        decoder_input = topi.detach()  # détaché pour teacher forcing

        loss += criterion(decoder_output, target_tensor[0, di].unsqueeze(0))
        if decoder_input.item() == EOS_token:
            break

    loss.backward()
    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length

# -----------------------------
# 6. Boucle d'entraînement
# -----------------------------
n_iters = 10000  # Augmentez (ex: 50000) pour de meilleurs résultats
print_every = 1000

print("Début de l'entraînement...")
total_loss = 0
for iter in range(1, n_iters + 1):
    pair = random.choice(training_pairs)
    input_tensor, target_tensor = pair
    loss = train_one_pair(input_tensor, target_tensor)
    total_loss += loss

    if iter % print_every == 0:
        print(f"Itération {iter} - Loss moyenne : {total_loss / print_every:.4f}")
        total_loss = 0

print("Entraînement terminé !")

# -----------------------------
# 7. Fonction d'évaluation / inférence
# -----------------------------
def evaluate(sentence, max_length=20):
    with torch.no_grad():
        input_tensor = tensor_from_sentence(lang, sentence)
        encoder_hidden = encoder(input_tensor)

        decoder_input = torch.tensor([[SOS_token]], device=device)
        decoder_hidden = encoder_hidden

        decoded_words = []
        for _ in range(max_length):
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
            topv, topi = decoder_output.topk(1)
            word_idx = topi.item()

            if word_idx == EOS_token:
                break
            else:
                decoded_words.append(lang.index2word[word_idx])

            decoder_input = topi.detach()

        return ' '.join(decoded_words)

# -----------------------------
# 8. Test interactif du chatbot
# -----------------------------
print("\n=== Chatbot prêt ! Tapez 'quit' pour arrêter ===")
while True:
    try:
        user_input = input("Vous : ").strip().lower()
        if user_input == 'quit':
            break
        if user_input == '':
            continue
        response = evaluate(user_input)
        print(f"Bot  : {response}")
    except KeyError:
        print("Bot  : Désolé, je ne connais pas certains mots de votre phrase.")
    except Exception as e:
        print(f"Bot  : Erreur : {e}")
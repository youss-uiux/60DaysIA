import torch
import torch.nn as nn
import torch.optim as optim
# import torch.nn.functional as F
import random
import json
import os
import glob
import re
import unicodedata

# -----------------------------
# 1. Chargement de fichiers JSON du dossier test (avec normalisation)
# -----------------------------
data_folder = "test"  # Le dossier que tu as copié
json_files = glob.glob(os.path.join(data_folder, "*.json"))

# Prendre seulement les 3 premiers fichiers (tu peux changer le nombre)
selected_files = json_files[:5]
print(f"Chargement de {len(selected_files)} fichiers JSON :")
for f in selected_files:
    print("  -", os.path.basename(f))

pairs = []

# normalisation : apostrophes standard + suppression des accents + nettoyage d'espaces
def normalize_text(s: str) -> str:
    s = s.lower().strip()
    # normaliser les apostrophes variantes
    s = s.replace("’", "'").replace("‘", "'").replace("`", "'")
    # remplacer multiples espaces
    s = re.sub(r"\s+", " ", s)
    # enlever accents (é -> e, etc.)
    s = ''.join(c for c in unicodedata.normalize('NFKD', s) if not unicodedata.combining(c))
    return s

for file_path in selected_files:
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)  # Chaque fichier contient une liste de dialogues
        for item in data:
            context = item.get("context", [])
            response = item.get("response", "").strip()
            if context and response and len(response.split()) > 1:  # réponse pas vide
                # On prend la DERNIÈRE phrase du context comme input
                input_sentence = normalize_text(context[-1].strip())
                target_sentence = normalize_text(response)
                if len(input_sentence.split()) > 1:  # évite les phrases trop courtes
                    pairs.append([input_sentence, target_sentence])

print(f"Total de {len(pairs)} paires extraites pour l'entraînement.\n")

# -----------------------------
# 2. Construction du vocabulaire (avec PAD + UNK)
# -----------------------------
SOS_token = 0
EOS_token = 1
UNK_token = 2
PAD_token = 3

# tokenisation moins agressive : garde les mots avec apostrophes intacts
def tokenize(sentence):
    # Garde les contractions (c'est, qu'est-ce, etc.) comme un seul token
    # Sépare uniquement la ponctuation finale (., ?, !, etc.)
    tokens = []
    # Découpe en mots en gardant les apostrophes
    words = re.findall(r"\b\w+(?:'\w+)*\b|[.!?;,]", sentence, re.UNICODE)
    for word in words:
        if word.strip():  # évite les tokens vides
            tokens.append(word)
    return tokens

class Lang:
    def __init__(self):
        self.word2index = {}
        self.word2count = {}
        self.index2word = {SOS_token: "SOS", EOS_token: "EOS", UNK_token: "UNK", PAD_token: "PAD"}
        self.n_words = 4

    def add_sentence(self, sentence):
        for word in tokenize(sentence):
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

print(f"Vocabulaire construit : {lang.n_words} mots uniques.\n")

# -----------------------------
# 3. Conversion phrase → indices et fonctions de batching/padding
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def indexes_from_sentence(lang, sentence):
    words = tokenize(sentence)
    indexes = [lang.word2index.get(word, UNK_token) for word in words]
    indexes.append(EOS_token)
    return indexes

from typing import List, Tuple

def make_batch(pairs_list: List[Tuple[str, str]], batch_size: int):
    # échantillonnage
    if batch_size <= 0:
        raise ValueError("batch_size doit être > 0")
    if len(pairs_list) == 0:
        raise ValueError("Aucune paire disponible pour créer un batch")
    if batch_size <= len(pairs_list):
        batch = random.sample(pairs_list, k=batch_size)
    else:
        batch = [random.choice(pairs_list) for _ in range(batch_size)]

    input_seqs = []
    target_seqs = []
    for inp, tgt in batch:
        input_seqs.append(indexes_from_sentence(lang, inp))
        target_seqs.append(indexes_from_sentence(lang, tgt))

    # padding
    max_in = max(len(s) for s in input_seqs)
    max_tgt = max(len(s) for s in target_seqs)

    padded_inputs = [s + [PAD_token] * (max_in - len(s)) for s in input_seqs]
    padded_targets = [s + [PAD_token] * (max_tgt - len(s)) for s in target_seqs]

    input_tensor = torch.tensor(padded_inputs, dtype=torch.long, device=device)  # (batch, max_in)
    target_tensor = torch.tensor(padded_targets, dtype=torch.long, device=device)  # (batch, max_tgt)
    return input_tensor, target_tensor

# training_pairs remain as raw text pairs (we'll batch on-the-fly)
# Division train/test pour validation
random.shuffle(pairs)
split_ratio = 0.8
split_index = int(len(pairs) * split_ratio)
training_pairs = pairs[:split_index]
validation_pairs = pairs[split_index:]
print(f"Division des données : {len(training_pairs)} pour l'entraînement, {len(validation_pairs)} pour la validation\n")

# -----------------------------
# 4. Modèles Encodeur et Décodeur (GRU) - inchangés mais compatibles batch
# -----------------------------
class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size, padding_idx=PAD_token)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)

    def forward(self, input_tensor):
        # input_tensor: (batch, seq_len)
        embedded = self.embedding(input_tensor)
        output, hidden = self.gru(embedded)  # output: (batch, seq, hidden), hidden: (1, batch, hidden)
        return output, hidden

class AttentionDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, max_length=50, dropout_p=0.1):
        super(AttentionDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.max_length = max_length
        self.dropout_p = dropout_p

        self.embedding = nn.Embedding(self.output_size, self.hidden_size, padding_idx=PAD_token)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size, batch_first=True)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input_tensor, hidden, encoder_outputs):
        # input_tensor: (batch, 1)
        # hidden: (1, batch, hidden_size)
        # encoder_outputs: (batch, seq_len, hidden_size)

        embedded = self.embedding(input_tensor)
        embedded = self.dropout(embedded)  # (batch, 1, hidden_size)

        batch_size = input_tensor.size(0)
        seq_len = encoder_outputs.size(1)

        # Calculer les poids d'attention
        # Concaténer l'embedding actuel avec le hidden state
        attn_weights = torch.cat((embedded.squeeze(1), hidden.squeeze(0)), 1)  # (batch, hidden_size * 2)

        # Adapter la dimension max_length à la longueur réelle de la séquence
        if seq_len <= self.max_length:
            attn_weights = self.attn(attn_weights)  # (batch, max_length)
            attn_weights = attn_weights[:, :seq_len]  # Garder seulement les positions valides
        else:
            # Si la séquence est plus longue, utiliser une projection adaptée
            attn_linear = nn.Linear(self.hidden_size * 2, seq_len).to(input_tensor.device)
            attn_weights = attn_linear(attn_weights)

        attn_weights = torch.softmax(attn_weights, dim=1).unsqueeze(1)  # (batch, 1, seq_len)

        # Appliquer l'attention aux sorties de l'encodeur
        attn_applied = torch.bmm(attn_weights, encoder_outputs)  # (batch, 1, hidden_size)

        # Combiner avec l'embedding
        output = torch.cat((embedded, attn_applied), 2)  # (batch, 1, hidden_size * 2)
        output = self.attn_combine(output)  # (batch, 1, hidden_size)

        output = torch.relu(output)
        output, hidden = self.gru(output, hidden)

        output = self.out(output.squeeze(1))  # (batch, output_size)
        return output, hidden, attn_weights.squeeze(1)

# -----------------------------
# 5. Paramètres et entraînement (mini-batchs)
# -----------------------------
hidden_size = 256
max_length = 50

encoder = EncoderRNN(lang.n_words, hidden_size).to(device)
decoder = AttentionDecoderRNN(hidden_size, lang.n_words, max_length).to(device)

# Optimiseurs et criterion (ignorer PAD dans la perte)
encoder_optimizer = optim.Adam(encoder.parameters(), lr=0.001)
decoder_optimizer = optim.Adam(decoder.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss(ignore_index=PAD_token)

# teacher forcing
teacher_forcing_ratio = 0.5

save_every = 2000  # sauvegarder le modèle toutes les N itérations
models_dir = os.path.join(os.path.dirname(__file__), 'models')
if not os.path.exists(models_dir):
    os.makedirs(models_dir, exist_ok=True)


def train_one_batch(input_tensor, target_tensor):
    # input_tensor: (batch, seq_in), target_tensor: (batch, seq_tgt)
    batch_size = input_tensor.size(0)
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    loss = torch.tensor(0.0, device=device)

    encoder_outputs, encoder_hidden = encoder(input_tensor)  # outputs: (batch, seq, hidden), hidden: (1, batch, hidden)

    decoder_input = torch.full((batch_size, 1), SOS_token, dtype=torch.long, device=device)
    decoder_hidden = encoder_hidden

    target_length = target_tensor.size(1)

    use_teacher = True if random.random() < teacher_forcing_ratio else False

    for di in range(target_length):
        decoder_output, decoder_hidden, _ = decoder(decoder_input, decoder_hidden, encoder_outputs)  # (batch, vocab)
        # target_tensor[:, di] -> shape (batch,)
        step_loss = criterion(decoder_output, target_tensor[:, di])
        loss = loss + step_loss

        if use_teacher:
            decoder_input = target_tensor[:, di].unsqueeze(1)  # (batch, 1)
        else:
            topi = decoder_output.argmax(dim=1).unsqueeze(1)  # (batch,1)
            decoder_input = topi.detach()

    loss.backward()
    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length

# -----------------------------
# 6. Inférence
# -----------------------------
def evaluate(sentence, max_length=30):
    with torch.no_grad():
        sentence = normalize_text(sentence)
        input_indexes = indexes_from_sentence(lang, sentence)
        input_tensor = torch.tensor([input_indexes], dtype=torch.long, device=device)
        encoder_outputs, encoder_hidden = encoder(input_tensor)

        decoder_input = torch.tensor([[SOS_token]], device=device)
        decoder_hidden = encoder_hidden

        decoded_words = []
        for _ in range(max_length):
            decoder_output, decoder_hidden, _ = decoder(decoder_input, decoder_hidden, encoder_outputs)
            topv, topi = decoder_output.topk(1)
            word_idx = topi.item()

            if word_idx == EOS_token:
                break
            decoded_words.append(lang.index2word.get(word_idx, "UNK"))

            decoder_input = topi.detach()

        response = ' '.join(decoded_words)

        # Filtrage des réponses trop courtes ou fragmentées
        if len(decoded_words) < 2:  # Réponse trop courte
            return "Je ne comprends pas bien."

        # Filtrer les réponses avec trop de tokens UNK
        unk_ratio = decoded_words.count("UNK") / len(decoded_words)
        if unk_ratio > 0.5:  # Plus de 50% de mots inconnus
            return "Pouvez-vous reformuler votre question ?"

        # Filtrer les réponses répétitives (même mot répété)
        if len(set(decoded_words)) <= 2 and len(decoded_words) > 3:
            return "Je ne sais pas quoi répondre à cela."

        return response

# Fonction de validation
def evaluate_on_validation_set():
    total_loss = 0.0
    num_batches = min(50, len(validation_pairs) // 8)  # Évaluer sur un sous-ensemble

    encoder.eval()
    decoder.eval()

    with torch.no_grad():
        for _ in range(num_batches):
            try:
                batch_size = min(8, len(validation_pairs))
                input_batch, target_batch = make_batch(validation_pairs, batch_size)

                # Calcul de la loss de validation (similaire à train_one_batch mais sans backprop)
                batch_size = input_batch.size(0)
                encoder_outputs, encoder_hidden = encoder(input_batch)
                decoder_input = torch.full((batch_size, 1), SOS_token, dtype=torch.long, device=device)
                decoder_hidden = encoder_hidden
                target_length = target_batch.size(1)

                batch_loss = 0.0
                for di in range(target_length):
                    decoder_output, decoder_hidden, _ = decoder(decoder_input, decoder_hidden, encoder_outputs)
                    step_loss = criterion(decoder_output, target_batch[:, di])
                    batch_loss += step_loss.item()
                    decoder_input = target_batch[:, di].unsqueeze(1)  # Teacher forcing pour validation

                total_loss += batch_loss / target_length
            except Exception as e:
                continue

    encoder.train()
    decoder.train()

    return total_loss / num_batches if num_batches > 0 else 0.0

# -----------------------------
# 7. Test interactif avec batching et sauvegarde de checkpoints
# -----------------------------
if __name__ == "__main__":
    n_iters = 20000  # Plus avec un vrai dataset
    print_every = 2000
    batch_size = 16

    print("Début de l'entraînement...\n")
    total_loss = 0.0

    for iter in range(1, n_iters + 1):
        if not training_pairs:
            print("Aucune paire d'entraînement disponible. Vérifiez les fichiers JSON dans le dossier 'test'.")
            break

        # préparer un mini-batch
        bsize = min(batch_size, max(1, len(training_pairs)))
        input_batch, target_batch = make_batch(training_pairs, bsize)

        loss = train_one_batch(input_batch, target_batch)
        total_loss += loss

        if iter % print_every == 0:
            avg_train_loss = total_loss / print_every
            val_loss = evaluate_on_validation_set()
            print(f"Itération {iter} - Loss entraînement : {avg_train_loss:.4f}, Loss validation : {val_loss:.4f}")

            # Early stopping simple : si la validation loss augmente trop
            if hasattr(evaluate_on_validation_set, 'best_val_loss'):
                if val_loss > evaluate_on_validation_set.best_val_loss * 1.1:  # 10% d'augmentation
                    print("Arrêt précoce détecté : la loss de validation augmente.")
            else:
                evaluate_on_validation_set.best_val_loss = val_loss

            if val_loss < getattr(evaluate_on_validation_set, 'best_val_loss', float('inf')):
                evaluate_on_validation_set.best_val_loss = val_loss

            total_loss = 0.0

        if iter % save_every == 0:
            # sauvegarde des checkpoints
            enc_path = os.path.join(models_dir, f"encoder_iter_{iter}.pt")
            dec_path = os.path.join(models_dir, f"decoder_iter_{iter}.pt")
            opt_enc = os.path.join(models_dir, f"encoder_opt_iter_{iter}.pt")
            opt_dec = os.path.join(models_dir, f"decoder_opt_iter_{iter}.pt")
            torch.save(encoder.state_dict(), enc_path)
            torch.save(decoder.state_dict(), dec_path)
            torch.save(encoder_optimizer.state_dict(), opt_enc)
            torch.save(decoder_optimizer.state_dict(), opt_dec)
            print(f"Checkpoint sauvegardé : itération {iter}")

    print("\nEntraînement terminé !\n")

    print("=== Chatbot prêt ! Tapez 'quit' pour arrêter ===\n")
    while True:
        try:
            user_input = input("Vous : ").strip()
            if user_input.lower() == 'quit':
                break
            if not user_input:
                continue
            response = evaluate(user_input)
            print(f"Bot  : {response}\n")
        except Exception as e:
            print(f"Bot  : Erreur : {e}\n")

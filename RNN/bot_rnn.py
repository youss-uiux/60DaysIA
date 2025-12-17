import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import re
from datasets import load_dataset

# Définir le dispositif (GPU si disponible)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Fonction pour charger le dataset Claire-Dialogue-French
def load_claire_dataset(max_pairs=1000):
    """Charge le dataset Claire et extrait les paires question-réponse"""
    try:
        print("Chargement du dataset Claire-Dialogue-French...")
        dataset = load_dataset("OpenLLM-France/Claire-Dialogue-French-0.1", split="train")

        pairs = []
        for item in dataset:
            if len(pairs) >= max_pairs:
                break

            # Le dataset Claire contient des conversations
            if "messages" in item:
                messages = item["messages"]
                for i in range(len(messages) - 1):
                    if (messages[i]["role"] == "user" and
                        messages[i+1]["role"] == "assistant"):

                        question = clean_text(messages[i]["content"])
                        response = clean_text(messages[i+1]["content"])

                        # Filtrer les phrases trop longues ou courtes
                        if (2 <= len(question.split()) <= 12 and
                            2 <= len(response.split()) <= 12 and
                            len(question) > 0 and len(response) > 0):
                            pairs.append([question, response])

        print(f"Dataset Claire chargé: {len(pairs)} paires de dialogue")
        return pairs

    except Exception as e:
        print(f"Erreur lors du chargement du dataset Claire: {e}")
        print("Utilisation des données d'exemple...")
        return [
            ["bonjour", "salut"],
            ["comment ca va", "bien et toi"],
            ["quel est ton nom", "je m'appelle assistant"],
            ["au revoir", "a bientot"],
            ["quel temps fait-il", "il pleut"],
            ["comment tu vas", "tres bien merci"],
            ["que fais tu", "je discute avec toi"],
            ["merci", "de rien"],
            ["bonne nuit", "dors bien"],
            ["aide moi", "bien sur je t aide"]
        ]

def clean_text(text):
    """Nettoie et normalise le texte"""
    # Convertir en minuscules
    text = text.lower()
    # Supprimer la ponctuation excessive et les caractères spéciaux
    text = re.sub(r'[^\w\s]', '', text)
    # Supprimer les espaces multiples
    text = ' '.join(text.split())
    return text.strip()

# Charger les données
pairs = load_claire_dataset(max_pairs=2000)

# Créer un vocabulaire
SOS_token = 0  # Start of sentence
EOS_token = 1  # End of sentence

class Lang:
    def __init__(self):
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2  # SOS et EOS

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

# Créer le langage partagé pour input et output
lang = Lang()
for pair in pairs:
    lang.add_sentence(pair[0])
    lang.add_sentence(pair[1])

# Fonction pour convertir une phrase en tenseur
def tensor_from_sentence(lang, sentence):
    indexes = [lang.word2index[word] for word in sentence.split(' ')]
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(1, -1)

# Préparer les tenseurs pour l'entraînement
training_pairs = [(tensor_from_sentence(lang, pair[0]), tensor_from_sentence(lang, pair[1])) for pair in pairs]

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)

    def forward(self, input_tensor):
        embedded = self.embedding(input_tensor)
        output, hidden = self.gru(embedded)
        return hidden  # On retourne seulement l'état caché final

    def init_hidden(self):
        return torch.zeros(1, 1, int(self.hidden_size), device=device)

class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input_tensor, hidden):
        embedded = self.embedding(input_tensor)
        output, hidden = self.gru(embedded, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden


def train(encoder, decoder, encoder_optimizer, decoder_optimizer, input_tensor, target_tensor, max_length=10):
    encoder_hidden = encoder.init_hidden()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_tensor.size(1)
    target_length = target_tensor.size(1)

    # Initialiser la perte comme un tenseur pour permettre backward
    loss = torch.tensor(0.0, device=device)

    # Encoder
    encoder_hidden = encoder(input_tensor)

    # Decoder
    decoder_input = torch.tensor([[SOS_token]], device=device)
    decoder_hidden = encoder_hidden

    for di in range(target_length):
        decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
        topv, topi = decoder_output.topk(1)
        # Ne pas écraser les dimensions batch/seq : garder topi avec shape (batch, seq_len)
        decoder_input = topi.detach()  # conserve la forme (1,1)

        # Accumuler la perte sans opération in-place
        loss = loss + F.nll_loss(decoder_output, target_tensor[0][di].unsqueeze(0))
        if decoder_input.item() == EOS_token:
            break

    loss.backward()
    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length

# Paramètres
hidden_size = 256
encoder = EncoderRNN(lang.n_words, hidden_size).to(device)
decoder = DecoderRNN(hidden_size, lang.n_words).to(device)

encoder_optimizer = optim.SGD(encoder.parameters(), lr=0.01)
decoder_optimizer = optim.SGD(decoder.parameters(), lr=0.01)

# Entraîner sur plus d'itérations pour tirer parti du dataset Claire
n_iters = 1000
print_every = 100
total_loss = 0

print(f"Début de l'entraînement sur {len(pairs)} paires de dialogue...")
print(f"Nombre d'itérations: {n_iters}")
print(f"Vocabulaire: {lang.n_words} mots")

for iter in range(1, n_iters + 1):
    training_pair = random.choice(training_pairs)
    input_tensor = training_pair[0]
    target_tensor = training_pair[1]

    loss = train(encoder, decoder, encoder_optimizer, decoder_optimizer, input_tensor, target_tensor)
    total_loss += loss

    if iter % print_every == 0:
        avg_loss = total_loss / print_every
        print(f'[{iter}/{n_iters}] Loss moyenne: {avg_loss:.4f}')
        total_loss = 0

print('Training finished')

# Fonction pour évaluer le modèle (génération de réponse)
def evaluate(encoder, decoder, sentence, max_length=10):
    """Génère une réponse pour une phrase donnée"""
    with torch.no_grad():
        # Préparer l'input
        input_tensor = tensor_from_sentence(lang, sentence)

        # Encoder
        encoder_hidden = encoder.init_hidden()
        encoder_hidden = encoder(input_tensor)

        # Decoder
        decoder_input = torch.tensor([[SOS_token]], device=device)
        decoder_hidden = encoder_hidden

        decoded_words = []

        for di in range(max_length):
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
            topv, topi = decoder_output.topk(1)

            if topi.item() == EOS_token:
                break
            else:
                decoded_words.append(lang.index2word[topi.item()])

            decoder_input = topi.detach()

        return ' '.join(decoded_words)

# Fonction pour tester le chatbot de manière interactive
def test_chatbot(encoder, decoder, test_pairs=None):
    """Test interactif du chatbot"""
    print("\n=== Test du Chatbot RNN ===")
    print("Tapez 'quit' ou 'exit' pour arrêter")

    # Test automatique avec quelques exemples
    if test_pairs:
        print("\nTest automatique:")
        for input_sentence, expected in test_pairs[:5]:
            try:
                response = evaluate(encoder, decoder, input_sentence)
                print(f"Input: {input_sentence}")
                print(f"Attendu: {expected}")
                print(f"Généré: {response}")
                print("-" * 30)
            except Exception as e:
                print(f"Erreur avec '{input_sentence}': {e}")

    # Test interactif
    print("\nTest interactif:")
    while True:
        try:
            user_input = input("\nVous: ").strip()

            if user_input.lower() in ['quit', 'exit', 'sortir', 'stop']:
                print("Au revoir!")
                break

            if not user_input:
                continue

            # Nettoyer l'input
            cleaned_input = clean_text(user_input)

            # Vérifier que tous les mots sont dans le vocabulaire
            unknown_words = [word for word in cleaned_input.split() if word not in lang.word2index]

            if unknown_words:
                print(f"Bot: Désolé, je ne connais pas ces mots: {unknown_words}")
                continue

            # Générer la réponse
            response = evaluate(encoder, decoder, cleaned_input)

            if response:
                print(f"Bot: {response}")
            else:
                print("Bot: Je ne sais pas quoi répondre...")

        except KeyboardInterrupt:
            print("\nAu revoir!")
            break
        except Exception as e:
            print(f"Erreur: {e}")
            print("Essayez avec des mots plus simples...")

# Sauvegarder le modèle entraîné
def save_model(encoder, decoder, lang, filepath="chatbot_rnn_model.pth"):
    """Sauvegarde le modèle entraîné"""
    torch.save({
        'encoder_state_dict': encoder.state_dict(),
        'decoder_state_dict': decoder.state_dict(),
        'lang': lang,
        'hidden_size': encoder.hidden_size,
        'vocab_size': lang.n_words
    }, filepath)
    print(f"Modèle sauvegardé dans {filepath}")

# Charger un modèle sauvegardé
def load_model(filepath="chatbot_rnn_model.pth"):
    """Charge un modèle sauvegardé"""
    checkpoint = torch.load(filepath, map_location=device)

    # Recréer les modèles
    encoder = EncoderRNN(checkpoint['vocab_size'], checkpoint['hidden_size']).to(device)
    decoder = DecoderRNN(checkpoint['hidden_size'], checkpoint['vocab_size']).to(device)

    # Charger les états
    encoder.load_state_dict(checkpoint['encoder_state_dict'])
    decoder.load_state_dict(checkpoint['decoder_state_dict'])

    return encoder, decoder, checkpoint['lang']

# Sauvegarder le modèle
save_model(encoder, decoder, lang)

# Tester le chatbot
print(f"\nVocabulaire créé: {lang.n_words} mots")
print("Exemples de mots:", list(lang.word2index.keys())[:10])

# Préparer quelques paires de test
test_pairs = pairs[:10] if len(pairs) > 10 else pairs

# Lancer le test
test_chatbot(encoder, decoder, test_pairs)


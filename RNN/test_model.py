# === CHARGER LE MEILLEUR CHECKPOINT (à ajouter après l'entraînement ou dans un nouveau script) ===
import os

import torch

from RNN.chatbot_francais_kaggle import encoder, decoder, encoder_optimizer, decoder_optimizer, evaluate

best_iter = 4000  # ou 4000 selon tes préférences
enc_path = f"models/encoder_iter_{best_iter}.pt"
dec_path = f"models/decoder_iter_{best_iter}.pt"
opt_enc_path = f"models/encoder_opt_iter_{best_iter}.pt"
opt_dec_path = f"models/decoder_opt_iter_{best_iter}.pt"

if os.path.exists(enc_path) and os.path.exists(dec_path):
    encoder.load_state_dict(torch.load(enc_path))
    decoder.load_state_dict(torch.load(dec_path))
    encoder_optimizer.load_state_dict(torch.load(opt_enc_path))
    decoder_optimizer.load_state_dict(torch.load(opt_dec_path))
    print(f"Modèle chargé depuis l'itération {best_iter} (meilleure généralisation)")
else:
    print("Checkpoint non trouvé ! Vérifie le dossier 'models'")

# Puis lance le test interactif
print("=== Chatbot prêt (meilleur modèle chargé) ! Tapez 'quit' pour arrêter ===\n")
while True:
    try:
        user_input = input("Vous : ").strip()
        if user_input.lower() == 'quit':
            break
        if not user_input:
            continue
        response = evaluate(user_input)
        print(f"Bot  : {response}\n")
    except KeyboardInterrupt:
        break
    except Exception as e:
        print(f"Bot  : Erreur : {e}\n")
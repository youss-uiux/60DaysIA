import numpy as np
import tensorflow as tf
import pickle
import re

from keras.src.ops import NotEqual
from tensorflow.keras import preprocessing

def load_model_components():
    """Charge tous les composants du modèle sauvegardé"""
    print("Chargement des composants du modèle...")

    # Charger le tokenizer
    with open('tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)

    # Charger les paramètres du modèle
    with open('model_params.pickle', 'rb') as handle:
        model_params = pickle.load(handle)

    # Charger les modèles d'inférence
    custom_objects = {
        'NotEqual': NotEqual
    }

    encoder_model = tf.keras.models.load_model(
        'encoder_model.h5',
        custom_objects=custom_objects
    )

    decoder_model = tf.keras.models.load_model(
        'decoder_model.h5',
        custom_objects=custom_objects
    )

    print("Modèle chargé avec succès!")
    return tokenizer, model_params, encoder_model, decoder_model

def preprocess_input(input_sentence, tokenizer, maxlen_questions):
    """Préprocesse l'entrée utilisateur"""
    input_sentence = re.sub('[^a-zA-ZàâäéèêëïîôöùûüÿçÀÂÄÉÈÊËÏÎÔÖÙÛÜŸÇ]', ' ', input_sentence.lower())
    tokens = input_sentence.lower().split()
    tokens_list = []
    for word in tokens:
        if word in tokenizer.word_index:
            tokens_list.append(tokenizer.word_index[word])
        else:
            tokens_list.append(tokenizer.word_index.get(word, 1))
    return preprocessing.sequence.pad_sequences([tokens_list], maxlen=maxlen_questions, padding='post')

def generate_response(input_text, tokenizer, model_params, encoder_model, decoder_model):
    """Génère une réponse du chatbot"""
    maxlen_questions = model_params['maxlen_questions']
    maxlen_answers = model_params['maxlen_answers']

    # Encoder l'entrée
    states_values = encoder_model.predict(preprocess_input(input_text, tokenizer, maxlen_questions), verbose=0)

    # Initialiser la séquence de décodage
    empty_target_seq = np.zeros((1, 1))
    empty_target_seq[0, 0] = tokenizer.word_index['start']

    stop_condition = False
    decoded_translation = ''

    while not stop_condition:
        dec_outputs, h, c = decoder_model.predict([empty_target_seq] + states_values, verbose=0)
        sampled_word_index = np.argmax(dec_outputs[0, -1, :])
        sampled_word = None

        for word, index in tokenizer.word_index.items():
            if sampled_word_index == index:
                decoded_translation += f' {word}'
                sampled_word = word
                break

        if sampled_word == 'end' or len(decoded_translation.split()) > maxlen_answers:
            stop_condition = True

        empty_target_seq = np.zeros((1, 1))
        empty_target_seq[0, 0] = sampled_word_index
        states_values = [h, c]

    # Nettoyer la réponse
    decoded_translation = decoded_translation.replace(' end', '').strip()
    return decoded_translation

def main():
    """Fonction principale pour tester le chatbot de manière interactive"""
    print("=" * 60)
    print("           TEST INTERACTIF DU CHATBOT")
    print("=" * 60)

    try:
        # Charger le modèle
        tokenizer, model_params, encoder_model, decoder_model = load_model_components()

        print(f"\nParamètres du modèle chargé:")
        print(f"- Longueur max des questions: {model_params['maxlen_questions']}")
        print(f"- Longueur max des réponses: {model_params['maxlen_answers']}")
        print(f"- Taille du vocabulaire: {model_params['VOCAB_SIZE']}")
        print("-" * 60)

        print("\n🤖 Chatbot prêt ! Entrez vos questions ci-dessous.")
        print("💡 Tapez 'quit', 'exit', 'sortir' ou 'arreter' pour quitter.")
        print("-" * 60)

        # Boucle interactive
        while True:
            print("\n", end="")
            user_input = input("👤 Vous: ")

            # Vérifier les commandes de sortie
            if user_input.lower().strip() in ['quit', 'exit', 'sortir', 'arreter', 'stop']:
                print("🤖 Bot: Au revoir ! À bientôt !")
                break

            # Vérifier que l'entrée n'est pas vide
            if not user_input.strip():
                print("🤖 Bot: Veuillez entrer une question s'il vous plaît.")
                continue

            # Générer et afficher la réponse
            try:
                response = generate_response(user_input, tokenizer, model_params, encoder_model, decoder_model)
                if response.strip():
                    print(f"🤖 Bot: {response}")
                else:
                    print("🤖 Bot: Je n'ai pas compris votre question. Pouvez-vous la reformuler ?")
            except Exception as e:
                print(f"🤖 Bot: Désolé, j'ai eu un problème pour traiter votre question. ({e})")

    except FileNotFoundError as e:
        print(f"\n❌ Erreur: Fichier manquant - {e}")
        print("\n📁 Assurez-vous que tous les fichiers du modèle sont présents:")
        print("   - tokenizer.pickle")
        print("   - model_params.pickle")
        print("   - encoder_model.h5")
        print("   - decoder_model.h5")
        print("\n💡 Exécutez d'abord 'chatbot.py' pour entraîner et sauvegarder le modèle.")
        print("💡 Assurez-vous que l'entraînement s'est terminé complètement sans erreur.")

    except Exception as e:
        print(f"\n❌ Erreur lors du chargement du modèle: {e}")
        print("💡 Vérifiez que le modèle a été correctement sauvegardé.")

if __name__ == "__main__":
    main()

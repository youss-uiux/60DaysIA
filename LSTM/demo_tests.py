"""
Exemple d'utilisation des tests pour le chatbot LSTM
Ce script montre comment utiliser les différents jeux de tests
"""

import numpy as np
from test_chatbot import get_tests, run_tests, evaluate_responses

def demo_tests():
    """
    Démonstration des différents types de tests disponibles
    """
    print("=== Démonstration des tests du chatbot ===\n")

    # Affichage de tous les types de tests disponibles
    test_types = ['basiques', 'ia', 'sujets', 'conversation', 'avances']

    for test_type in test_types:
        tests = get_tests(test_type)
        print(f"🧪 Tests {test_type.upper()} ({len(tests)} questions):")
        for i, test in enumerate(tests, 1):
            print(f"   {i}. {test}")
        print()

def example_integration_with_chatbot():
    """
    Exemple d'intégration avec le chatbot LSTM
    """
    print("=== Exemple d'intégration ===")
    print("""
    # Dans votre fichier chatbot.py, vous pouvez maintenant utiliser:
    
    from test_chatbot import get_tests
    
    # Pour les tests de base
    tests = get_tests('basiques')
    
    # Pour les tests sur l'IA
    tests = get_tests('ia')
    
    # Pour tester différents sujets
    tests = get_tests('sujets')
    
    # Pour des conversations plus naturelles
    tests = get_tests('conversation')
    
    # Pour des tests plus avancés
    tests = get_tests('avances')
    """)

def simulate_chatbot_responses():
    """
    Simulation des réponses du chatbot pour démonstration
    """
    print("=== Simulation de test avec réponses ===\n")

    # Réponses simulées pour la démonstration
    def fake_chatbot(question):
        responses = {
            'Bonjour': 'Salut ! Comment ça va ?',
            'Es-tu un robot': 'Oui, je suis une intelligence artificielle.',
            'Quel est ton nom': 'Je suis un chatbot créé avec LSTM.',
            'Comment ça va': 'Ça va bien, merci de demander !',
            'à bientôt': 'Au revoir ! À bientôt !'
        }
        return responses.get(question, 'Je ne comprends pas bien votre question.')

    # Test avec les réponses simulées
    results = run_tests('basiques', fake_chatbot)

    # Évaluation des résultats
    metrics = evaluate_responses(results)

    print("\n=== Métriques d'évaluation ===")
    print(f"Tests réussis: {metrics['successful_tests']}/{metrics['total_tests']}")
    print(f"Taux de réussite: {metrics['success_rate']:.1f}%")
    print(f"Longueur moyenne des réponses: {metrics['average_response_length']:.1f} mots")
    if metrics['failed_tests']:
        print(f"Tests échoués: {metrics['failed_tests']}")

if __name__ == "__main__":
    demo_tests()
    example_integration_with_chatbot()
    simulate_chatbot_responses()

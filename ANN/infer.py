import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import sys

# Même transformation que pour l'entraînement
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


def load_model(model_path: str, device: torch.device = None, num_classes: int = 5, pretrained: bool = False):
    """Charge un ResNet18 et restaure les poids sauvegardés.

    Args:
        model_path: chemin vers le .pth sauvegardé
        device: torch.device ou None (détecte automatiquement)
        num_classes: nombre de classes du classifieur final
        pretrained: bool, si True essaie de charger les poids pré-entraînés (par défaut False)

    Returns:
        modèle PyTorch en mode evaluation
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = models.resnet18(pretrained=pretrained)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)

    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def predict_image(image_path: str, model, device: torch.device = None, topk: int = 3):
    """Prédit les indices de classes et probabilités pour une image.
    Retourne une liste de tuples (classe_index, prob).
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    img = Image.open(image_path).convert('RGB')
    x = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(x)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        top_probs, top_idx = probs.topk(topk, dim=1)

    results = []
    for p, idx in zip(top_probs[0].tolist(), top_idx[0].tolist()):
        results.append((int(idx), float(p)))
    return results


if __name__ == '__main__':
    # Usage simple depuis la ligne de commande : python -m ANN.infer <image_path> [model_path]
    if len(sys.argv) < 2:
        print('Usage: python -m ANN.infer <image_path> [model_path]')
        sys.exit(1)

    image_path = sys.argv[1]
    model_path = sys.argv[2] if len(sys.argv) > 2 else 'mon_classifieur_fleurs.pth'

    model = load_model(model_path, pretrained=False)
    preds = predict_image(image_path, model, topk=5)
    print('Prédictions (classe_index, prob):')
    for idx, prob in preds:
        print(f'  {idx}: {prob:.4f}')


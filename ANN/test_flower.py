# test_fleur.py  ← VERSION CORRIGÉE (plus de fichier manquant !)
import torch
from torchvision import transforms, models
import torch.nn as nn
from PIL import Image
import matplotlib.pyplot as plt
from flowers_dataset import FlowersDataset  # ton wrapper

# 1. Chargement du modèle
model = models.resnet50(weights=None)  # weights=None pour éviter le warning
model.fc = nn.Linear(2048, 102)
model.load_state_dict(torch.load("models/flowers102_resnet50.pth"))
model.eval()

# 2. Transformations (identiques au val/test)
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 3. Chargement des noms de classes depuis le dataset lui-même
# (ça marche car tu as déjà téléchargé les données)
flower_ds = FlowersDataset(root='./flowers', split='test', download=False, transform=None)
class_names = flower_ds.dataset.classes  # liste de 102 noms : ['pink primrose', 'hard-leaved pocket orchid', ...]


# ou si ConcatDataset : class_names = flower_ds.dataset.datasets[0].classes

# 4. Prédiction simple
def predict_flower(image_path):
    img = Image.open(image_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0)

    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted_idx = torch.max(probabilities, 1)

    predicted_class = class_names[predicted_idx.item()]
    confidence_percent = confidence.item() * 100

    plt.figure(figsize=(8, 6))
    plt.imshow(img)
    plt.axis('off')
    plt.title(f"Prédiction : {predicted_class}\nConfiance : {confidence_percent:.1f}%",
              fontsize=16, color="green" if confidence_percent > 70 else "orange")
    plt.show()

    print(f"→ C'est un(e) : {predicted_class}")
    print(f"→ Confiance : {confidence_percent:.1f}%")


# 5. Top 5 prédictions (plus fun et réaliste)
def predict_flower_top5(image_path):
    img = Image.open(image_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0)

    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = torch.softmax(outputs, dim=1).squeeze()
        top5_prob, top5_idx = torch.topk(probabilities, 5)

    plt.figure(figsize=(10, 7))
    plt.imshow(img)
    plt.axis('off')

    title = "Top 5 prédictions :\n"
    for i in range(5):
        class_name = class_names[top5_idx[i].item()]
        conf = top5_prob[i].item() * 100
        title += f"{i + 1}. {class_name} ({conf:.1f}%)\n"

    plt.title(title.strip(), fontsize=14)
    plt.show()


# ===== TESTE ICI =====
# Crée un dossier test_images/ et mets-y une photo de fleur
predict_flower_top5("lol.jpeg")
# predict_flower("test_images/tulip.jpg")
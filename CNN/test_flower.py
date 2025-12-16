# test_fleur.py  ← VERSION FINALE (console only + image propre)
import torch
from torchvision import transforms, models
import torch.nn as nn
from PIL import Image
import matplotlib.pyplot as plt
from flowers_dataset import FlowersDataset
import os

# ==================== 1. Modèle ====================
model = models.resnet50(weights=None)
model.fc = nn.Linear(2048, 102)
model.load_state_dict(torch.load("models/flowers102_resnet50.pth", map_location="cpu"))
model.eval()

# ==================== 2. Transform ====================
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ==================== 3. Noms des 102 fleurs (robuste) ====================
try:
    # Méthode la plus propre avec ton wrapper
    ds = FlowersDataset(root='./flowers', split='test', download=False, transform=None)
    if hasattr(ds.dataset, 'classes'):
        class_names = ds.dataset.classes
    elif hasattr(ds.dataset, 'datasets'):  # ConcatDataset
        class_names = ds.dataset.datasets[0].classes
    else:
        raise Exception()
except:
    print("Impossible de charger les noms automatiquement → on charge depuis labels.json")
    import json
    with open("./flowers/labels.json", "r") as f:
        data = json.load(f)
    # Les clés sont "1", "2", ..., on les remet de 0 à 101
    class_names = [data[str(i+1)] for i in range(102)]

print(f"{len(class_names)} classes chargées ✓ (ex: {class_names[0]}, {class_names[1]}, ..., {class_names[-1]})")

# ==================== 4. Prédiction avec affichage CONSOLE ONLY ====================
def predict(image_path):
    if not os.path.exists(image_path):
        print(f"ERREUR : Image non trouvée → {image_path}")
        return

    img = Image.open(image_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0)

    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.softmax(outputs, dim=1).squeeze()
        top5_prob, top5_idx = torch.topk(probs, 5)

    print("\n" + "="*60)
    print(f" PHOTO : {os.path.basename(image_path)}")
    print("="*60)
    for i in range(5):
        idx = top5_idx[i].item()
        name = class_names[idx]
        conf = top5_prob[i].item() * 100
        star = " ← MEILLEURE PRÉDICTION" if i == 0 else ""
        print(f"{i+1:2d}. {name:<35} → {conf:6.2f}%{star}")
    print("="*60)

    # Affichage propre de l'image SANS texte dessus
    plt.figure(figsize=(8, 8))
    plt.imshow(img)
    plt.axis('off')
    plt.title(os.path.basename(image_path), fontsize=16, pad=20)
    plt.show()

# ==================== TESTE TES PHOTOS ICI ====================
# Change juste le nom du fichier ou mets plusieurs lignes

predict("lol.jpeg")
predict("Gazania.jpeg")
predict("Oxeye-Daisy.jpeg")
predict("tournesol.jpeg")

print("\nPrêt ! Lance ce script et regarde la magie dans la console")
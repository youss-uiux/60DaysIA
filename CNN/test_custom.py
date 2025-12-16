# python
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms
import os
from CNN.flowers_dataset import FlowersDataset

device = "cuda" if torch.cuda.is_available() else "cpu"

class CNN(nn.Module):
    def __init__(self, in_channels, num_classes, input_size=(3, 224, 224)):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=8, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, padding=1)

        # déterminer dynamiquement la taille aplatie après les conv/pool
        with torch.no_grad():
            dummy = torch.zeros(1, *input_size)
            x = F.relu(self.conv1(dummy))
            x = self.pool(x)
            x = F.relu(self.conv2(x))
            x = self.pool(x)
            self._to_linear = int(x.numel() // x.shape[0])

        self.fc1 = nn.Linear(self._to_linear, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x

# chemins (adapter si nécessaire)
ckpt_path = os.path.join("checkpoints", "flowers_cnn.pth")
image_path = "iris.jpeg"

# chargement du checkpoint
if not os.path.exists(ckpt_path):
    raise FileNotFoundError(f"Checkpoint introuvable: `{ckpt_path}`")

checkpoint = torch.load(ckpt_path, map_location=device)
num_classes = checkpoint.get("num_classes", 102)
in_channels = checkpoint.get("in_channels", 3)

# Récupérer les noms de classes depuis le dataset (fallback possible)
try:
    ds = FlowersDataset(root='./flowers', download=False, transform=None)
    class_names = ds.classes if getattr(ds, 'classes', None) is not None else None
except Exception:
    class_names = None

if class_names is None:
    # fallback : générer des noms génériques
    class_names = [f"class_{i}" for i in range(num_classes)]

# instanciation et chargement des poids
model = CNN(in_channels=in_channels, num_classes=num_classes).to(device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# même transform que pour l'entraînement
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# lecture et prédiction
if not os.path.exists(image_path):
    raise FileNotFoundError(f"Image introuvable: `{image_path}`")

img = Image.open(image_path).convert("RGB")
img_tensor = transform(img).unsqueeze(0).to(device)

with torch.no_grad():
    outputs = model(img_tensor)
    pred = torch.argmax(outputs, dim=1).item()

# afficher le nom de la fleur si disponible
flower_name = class_names[pred] if pred < len(class_names) else f"index_{pred}"
print("Classe prédite :", pred)
print("Nom de la fleur prédite :", flower_name)

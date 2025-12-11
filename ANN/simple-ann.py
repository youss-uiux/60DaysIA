# ann_simple_sur_images.py  ← À lancer pour voir le désastre (et c'est génial !)
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from flowers_dataset import FlowersDataset
from torch.utils.data import DataLoader
from tqdm import tqdm

# ==================== 1. Dataset (images aplaties) ====================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),  # → image devient un tenseur 3x224x224
])

train_ds = FlowersDataset(root='./flowers', split='train', transform=transform)
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)


# ==================== 2. ANN simple (MLP) – ZÉRO convolution ====================
class SimpleANN(nn.Module):
    def __init__(self, input_size=3 * 224 * 224, num_classes=102):
        super().__init__()
        self.network = nn.Sequential(
            nn.Flatten(),  # → 3 × 224 × 224 = 150 528 valeurs
            nn.Linear(input_size, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)  # 102 fleurs
        )

    def forward(self, x):
        return self.network(x)


# ==================== 3. Modèle + optimiseur ====================
model = SimpleANN()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ==================== 4. Entraînement (20 epochs) ====================
print("Début de l'entraînement du ANN simple sur images brutes...")
for epoch in range(20):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}/20"):
        images = images.to(device)  # 32 × 3 × 224 × 224
        labels = labels.to(device)

        # Aplatir les images → le réseau ne voit plus que des pixels
        # (pas de notion de voisinage, de forme, etc.)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    acc = 100. * correct / total
    print(f"Epoch {epoch + 1:2d} → Loss: {running_loss / len(train_loader):.4f} | Accuracy: {acc:.2f}%")

# ==================== 5. Sauvegarde ====================
torch.save(model.state_dict(), "models/ann_simple_flowers.pth")
print("\nModèle ANN simple sauvegardé → models/ann_simple_flowers.pth")
print("Résultat typique : 5 à 15% d'accuracy max → c'est NORMAL !")
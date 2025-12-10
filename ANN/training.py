# train_flowers102.py
import torch
from torch.utils.data import DataLoader
from torchvision import transforms, models
import torch.nn as nn
import torch.optim as optim
from flowers_dataset import FlowersDataset
from tqdm import tqdm
import os

# Transforms
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Chargement avec splits corrects
train_ds = FlowersDataset(root='./flowers', split='train', download=False, transform=train_transform)
val_ds = FlowersDataset(root='./flowers', split='val', download=False, transform=val_transform)
test_ds = FlowersDataset(root='./flowers', split='test', download=False, transform=val_transform)

print(f"Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)
test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)

# Modèle
model = models.resnet50(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, 102)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=0.001)

# Entraînement simple (phase 1)
for epoch in range(15):
    model.train()
    running_loss = 0.0
    for images, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}/15"):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    print(f"Epoch {epoch + 1} - Loss: {running_loss / len(train_loader):.4f}")

# Sauvegarde
os.makedirs("models", exist_ok=True)
torch.save(model.state_dict(), "models/flowers102_resnet50.pth")
print("Modèle entraîné et sauvegardé !")
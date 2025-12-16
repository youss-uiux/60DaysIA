import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


import torch
from torch import optim
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# !pip install torchvision
import torchvision

import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms

# !pip install torchmetrics
import torchmetrics

from CNN.flowers_dataset import FlowersDataset

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])


batch_size = 32

dataset = FlowersDataset(root='./flowers', download=True, transform=transform)
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)


def imshow(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

# Visualiser quelques images
dataiter = iter(train_loader)
images, labels = next(dataiter)
imshow(torchvision.utils.make_grid(images))

class CNN(nn.Module):

    def __init__(self, in_channels, num_classes, input_size=(3, 224, 224)):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=8, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, padding=1)


        with torch.no_grad():
            dummy = torch.zeros(1, *input_size)
            x = F.relu(self.conv1(dummy))
            x = self.pool(x)
            x = F.relu(self.conv2(x))
            x = self.pool(x)
            self._to_linear = x.numel() // x.shape[0]

        self.fc1 = nn.Linear(self._to_linear, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x


device = "cuda" if torch.cuda.is_available() else "cpu"

model = CNN(in_channels=3, num_classes=102).to(device)
print(model)

# Define the loss function
criterion = nn.CrossEntropyLoss()

# Define the optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)


num_epochs = 10
for epoch in range(num_epochs):
    print(f"Epoch [{epoch+1}/{num_epochs}]")
    model.train()
    for data, targets in tqdm(train_loader):
        data, targets = data.to(device), targets.to(device)
        scores = model(data)
        loss = criterion(scores, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


# Set up of multiclass accuracy metric
from torchmetrics import Accuracy, Precision, Recall

acc = Accuracy(task="multiclass",num_classes=10)
precision = Precision(task="multiclass", num_classes=10)
recall = Recall(task="multiclass", num_classes=10)

# Iterate over the dataset batches
model.eval()
with torch.no_grad():
   for images, labels in test_loader:
       # Get predicted probabilities for test data batch
       outputs = model(images)
       _, preds = torch.max(outputs, 1)
       acc(preds, labels)
       precision(preds, labels)
       recall(preds, labels)

#Compute total test accuracy
test_accuracy = acc.compute()
print(f"Test accuracy: {test_accuracy}")
import os
os.makedirs("checkpoints", exist_ok=True)

# Sauvegarde du modèle
torch.save({
    "model_state_dict": model.state_dict(),
    "num_classes": 102,
    "in_channels": 3
}, "checkpoints/flowers_cnn.pth")

print("✅ Modèle sauvegardé : checkpoints/flowers_cnn.pth")
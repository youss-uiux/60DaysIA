from flowers_dataset import FlowersDataset
import matplotlib.pyplot as plt

# Test du dataset
if __name__ == "__main__":
    # Essaie d'abord avec Flowers102 intégré à torchvision (recommandé)
    dataset = FlowersDataset(
        root='./flowers',     # dossier où les données seront téléchargées
        download=True,        # télécharge automatiquement si Flowers102 est dispo
        transform=None        # on met None pour voir les images brutes
    )

    print(f"Nombre total d'images : {len(dataset)}")
    print(f"Nombre de classes : {len(dataset.dataset.classes) if hasattr(dataset.dataset, 'classes') else 'inconnu'}")

    # Afficher les 8 premières images
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    axes = axes.ravel()

    for i in range(8):
        img, label = dataset[i]
        axes[i].imshow(img)
        # Si on utilise ImageFolder, label est un int → on récupère le nom de classe
        if hasattr(dataset.dataset, 'classes'):
            title = dataset.dataset.classes[label] if hasattr(dataset.dataset, 'class_to_idx') else str(label)
        else:
            title = str(label)
        axes[i].set_title(title)
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()
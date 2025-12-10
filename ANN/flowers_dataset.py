# flowers_dataset.py  ← VERSION CORRIGÉE ET COMPLÈTE (à remplacer entièrement)
from torch.utils.data import Dataset, ConcatDataset
import torchvision.datasets as datasets
from torchvision.datasets import ImageFolder

# On essaie d'importer Flowers102 proprement
Flowers102 = None
try:
    from torchvision.datasets import Flowers102  # torchvision >= 0.15
except ImportError:
    pass

class FlowersDataset(Dataset):
    """
    Wrapper intelligent pour Oxford 102 Flower Dataset.
    Fonctionne dans tous les cas :
    - Avec torchvision récent → utilise Flowers102 + splits train/val/test
    - Avec torchvision ancien → fallback sur ImageFolder
    """
    def __init__(self, root='./flowers', split=None, download=False, transform=None, target_transform=None):
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.split = split  # 'train', 'val', 'test' ou None (tout)

        if Flowers102 is not None and split is not None:
            # Cas idéal : on a Flowers102 + on veut un split spécifique
            try:
                self.dataset = Flowers102(
                    root=root,
                    split=split,
                    download=download,
                    transform=transform,
                    target_transform=target_transform
                )
                return
            except Exception as e:
                print(f"Erreur avec Flowers102(split={split}): {e}. Fallback sur ImageFolder.")

        if Flowers102 is not None and split is None:
            # On veut TOUTES les données (concaténation des 3 splits)
            datasets = []
            for s in ['train', 'val', 'test']:
                try:
                    ds = Flowers102(root=root, split=s, download=download,
                                    transform=transform, target_transform=target_transform)
                    datasets.append(ds)
                except Exception:
                    continue
            if datasets:
                self.dataset = ConcatDataset(datasets)
                return

        # Fallback final : ImageFolder (si Flowers102 n'existe pas ou échoue)
        if download:
            print("Attention : Flowers102 non disponible ou échec → utilisez ImageFolder manuellement.")
        self.dataset = ImageFolder(root=root, transform=transform, target_transform=target_transform)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

    # Utile pour connaître les classes
    @property
    def classes(self):
        if hasattr(self.dataset, 'classes'):
            return self.dataset.classes
        elif hasattr(self.dataset, 'datasets'):  # ConcatDataset
            return self.dataset.datasets[0].classes
        else:
            return getattr(self.dataset, 'classes', None)
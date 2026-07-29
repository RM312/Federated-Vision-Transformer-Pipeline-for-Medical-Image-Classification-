import os
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader

from config import IMG_SIZE, BATCH_SIZE

####################################################
# DATA TRANSFORMS
####################################################

train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )
])

test_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )
])

####################################################
# CHANGE THIS PATH
####################################################

DATASET_ROOT = "client_data"

####################################################
# CLIENT PATHS
####################################################

client_dirs = [
    "ClientA",
    "ClientB"
]

client_train_loaders = []
client_val_loaders = []
client_test_loaders = []

for client in client_dirs:

    train_path = os.path.join(
        DATASET_ROOT,
        client,
        "train"
    )

    val_path = os.path.join(
        DATASET_ROOT,
        client,
        "val"
    )

    test_path = os.path.join(
        DATASET_ROOT,
        client,
        "test"
    )

    train_dataset = datasets.ImageFolder(
        train_path,
        transform=train_transform
    )

    val_dataset = datasets.ImageFolder(
        val_path,
        transform=test_transform
    )

    test_dataset = datasets.ImageFolder(
        test_path,
        transform=test_transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    client_train_loaders.append(train_loader)
    client_val_loaders.append(val_loader)
    client_test_loaders.append(test_loader)

print(f"{len(client_train_loaders)} clients loaded.")

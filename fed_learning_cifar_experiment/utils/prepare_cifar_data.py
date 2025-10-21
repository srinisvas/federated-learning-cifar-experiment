"""
prepare_cifar_data.py
=====================
Utility script to pre-download and cache the CIFAR-10 dataset locally.

Run this once (with internet access). It will store data under ./data/
so your federated learning app can later run fully offline.
"""

from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor
from pathlib import Path

def prepare_cifar10(data_dir="./data"):
    """Download CIFAR-10 train and test datasets to local storage."""
    data_root = Path(data_dir)
    data_root.mkdir(parents=True, exist_ok=True)

    print(f"Downloading CIFAR-10 dataset to: {data_root.resolve()}")

    # Simple transform, just to make sure dataset is compatible with later usage
    transform = ToTensor()

    # Download training data
    print("Downloading training set...")
    CIFAR10(root=data_root, train=True, download=True, transform=transform)

    # Download test data
    print("Downloading test set...")
    CIFAR10(root=data_root, train=False, download=True, transform=transform)

    print("CIFAR-10 dataset is ready for offline use!")

if __name__ == "__main__":
    prepare_cifar10()


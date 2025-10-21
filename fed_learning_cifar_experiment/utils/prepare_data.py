"""
prepare_data.py — One-time script to download CIFAR-10 for offline federated learning.
"""

import os
from pathlib import Path
from datasets import load_dataset
from torchvision.datasets import CIFAR10

DATA_DIR = Path("./data")
HF_DATASET_DIR = DATA_DIR / "cifar10_hf"
TORCH_DATASET_DIR = DATA_DIR / "cifar10_torch"

def ensure_dirs():
    HF_DATASET_DIR.mkdir(parents=True, exist_ok=True)
    TORCH_DATASET_DIR.mkdir(parents=True, exist_ok=True)

def download_hf_cifar10():
    print(" Downloading CIFAR-10 from Hugging Face…")
    ds = load_dataset("uoft-cs/cifar10")
    ds.save_to_disk(str(HF_DATASET_DIR))
    print(f" Hugging Face dataset saved to {HF_DATASET_DIR}")

def download_torchvision_cifar10():
    print(" Downloading CIFAR-10 from TorchVision…")
    CIFAR10(root=str(TORCH_DATASET_DIR), train=True, download=True)
    CIFAR10(root=str(TORCH_DATASET_DIR), train=False, download=True)
    print(f" TorchVision dataset saved to {TORCH_DATASET_DIR}")

if __name__ == "__main__":
    ensure_dirs()
    download_hf_cifar10()
    download_torchvision_cifar10()
    print(" All datasets are ready for offline use.")

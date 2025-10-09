"""fed-learning-cifar-experiment: A Flower / PyTorch app."""

from collections import OrderedDict

import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, List

import numpy as np
from torch.utils.data import DataLoader, Subset, random_split
from torchvision.datasets import CIFAR10
from torchvision.transforms import Compose, Normalize, ToTensor

from fed_learning_cifar_experiment.utils.backdoor_attack import collate_with_backdoor
from fed_learning_cifar_experiment.models.basic_cnn_model import Net
from fed_learning_cifar_experiment.models.resnet_cnn_model import tiny_resnet18

_cifar_train_dataset = None
_partition_cache: Dict[tuple, List[List[int]]] = {}
_rng = np.random.default_rng(42)

def get_resnet_cnn_model(num_classes: int = 10) -> nn.Module:
    return tiny_resnet18(num_classes=num_classes, base_width=8)

def get_basic_cnn_model() -> nn.Module:
    return Net()

def _load_cifar10_dataset(transform):
    """Load the CIFAR10 training dataset with torchvision, falling back to local data."""
    data_root = Path("./data")
    try:
        return CIFAR10(root=data_root, train=True, download=True, transform=transform)
    except Exception:
        return CIFAR10(root=data_root, train=True, download=False, transform=transform)


def _create_dirichlet_partitions(targets: List[int], num_partitions: int, alpha_val: float) -> List[List[int]]:
    """Create Dirichlet partitions over the dataset labels."""
    labels = np.array(targets)
    classes = np.unique(labels)
    partitions: List[List[int]] = [[] for _ in range(num_partitions)]

    for cls in classes:
        cls_indices = np.where(labels == cls)[0]
        _rng.shuffle(cls_indices)
        split_points = (np.cumsum(_rng.dirichlet(np.full(num_partitions, alpha_val))) * len(cls_indices)).astype(int)[:-1]
        cls_split = np.split(cls_indices, split_points)
        for partition_idx, subset in enumerate(cls_split):
            partitions[partition_idx].extend(subset.tolist())

    for indices in partitions:
        _rng.shuffle(indices)

    return partitions


def _get_partition_indices(num_partitions: int, alpha_val: float) -> List[List[int]]:
    global _partition_cache, _cifar_train_dataset
    cache_key = (num_partitions, float(alpha_val))
    if cache_key not in _partition_cache:
        transform = Compose(
            [ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        )
        if _cifar_train_dataset is None:
            _cifar_train_dataset = _load_cifar10_dataset(transform)
        _partition_cache[cache_key] = _create_dirichlet_partitions(_cifar_train_dataset.targets, num_partitions, alpha_val)
    return _partition_cache[cache_key]


def load_data(partition_id: int, num_partitions: int, alpha_val: float, backdoor_enabled: bool = False,
              target_label: int = 2, poison_fraction: float = 0.1):
    """Load partition CIFAR10 data."""
    global _cifar_train_dataset
    partitions = _get_partition_indices(num_partitions, alpha_val)
    indices = partitions[partition_id]
    partition_subset = Subset(_cifar_train_dataset, indices)

    partition_size = len(partition_subset)
    if partition_size == 0:
        empty_train_loader = DataLoader(partition_subset, batch_size=64, shuffle=True)
        empty_test_loader = DataLoader(partition_subset, batch_size=64)
        return empty_train_loader, empty_test_loader

    train_len = int(partition_size * 0.8)
    test_len = partition_size - train_len
    if train_len == 0 and partition_size > 0:
        train_len = 1 if partition_size > 1 else partition_size
        test_len = partition_size - train_len
    if test_len == 0 and partition_size > 1:
        test_len = 1
        train_len = partition_size - test_len

    train_subset, test_subset = random_split(
        partition_subset,
        [train_len, test_len],
        generator=torch.Generator().manual_seed(42),
    )

    if backdoor_enabled:
        training_data = DataLoader(
            train_subset,
            batch_size=64,
            shuffle=True,
            collate_fn=lambda batch: collate_with_backdoor(batch, num_backdoor_per_batch=20, target_label=target_label)
        )
        test_data = DataLoader(test_subset, batch_size=64)
    else:
        training_data = DataLoader(train_subset, batch_size=64, shuffle=True)
        test_data = DataLoader(test_subset, batch_size=64)

    return training_data, test_data


def train(net, training_data, epochs, device, lr=0.1):
    """Train the model on the training set."""
    net.to(device)  # move model to GPU if available
    criterion = torch.nn.CrossEntropyLoss().to(device)
    #optimizer = torch.optim.Adam(net.parameters(), lr=0.1)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    net.train()
    running_loss = 0.0
    for _ in range(epochs):
        for batch in training_data:
            if isinstance(batch, dict):
                images = batch["img"]
                labels = batch["label"]
            else:
                images, labels = batch
            optimizer.zero_grad()
            loss = criterion(net(images.to(device)), labels.to(device))
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

    avg_training_loss = running_loss / len(training_data)
    return avg_training_loss


def test(net, test_data, device):
    """Validate the model on the test set."""
    net.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    correct, loss = 0, 0.0
    with torch.no_grad():
        for batch in test_data:
            if isinstance(batch, dict):
                images = batch["img"].to(device)
                labels = batch["label"].to(device)
            else:
                images, labels = batch
                images = images.to(device)
                labels = labels.to(device)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
    accuracy = correct / len(test_data.dataset)
    loss = loss / len(test_data)
    return loss, accuracy


def get_weights(net):
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


def set_weights(net, parameters):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)

def load_test_data_for_eval(batch_size=64):
    """Load CIFAR10 data."""

    pytorch_transforms = Compose(
        [ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    data_root = Path("./data")
    try:
        test_dataset = CIFAR10(root=data_root, train=False, download=True, transform=pytorch_transforms)
    except Exception:
        test_dataset = CIFAR10(root=data_root, train=False, download=False, transform=pytorch_transforms)

    test_data = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return test_data

def test_eval(net, test_data, device):
    """Evaluate the updated model on the test set for evaluations."""
    net.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    correct, loss = 0, 0.0
    with torch.no_grad():
        for batch in test_data:
            if isinstance(batch, dict):
                images, labels = batch["img"], batch["label"]
            else:
                images, labels = batch
            images = images.to(device)
            labels = labels.to(device)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
    accuracy = correct / len(test_data.dataset)
    loss = loss / len(test_data)
    return loss, accuracy

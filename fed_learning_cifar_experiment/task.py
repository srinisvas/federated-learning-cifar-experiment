"""fed-learning-cifar-experiment: A Flower / PyTorch app."""
import os
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from torchvision.datasets import CIFAR10
from torchvision.transforms import RandomCrop, RandomHorizontalFlip, ColorJitter
from torchvision.transforms import Compose, Normalize, ToTensor
from torchvision.transforms import v2
from datasets import load_from_disk, DatasetDict

from fed_learning_cifar_experiment.utils.backdoor_attack import collate_with_backdoor
from fed_learning_cifar_experiment.models.basic_cnn_model import Net
from fed_learning_cifar_experiment.models.resnet_cnn_model import tiny_resnet18
from fed_learning_cifar_experiment.utils.drichlet_partition import dirichlet_indices

fds = None  # Cache FederatedDataset
dirichlet_cache = None

base_dir = os.path.dirname(__file__)
local_torch_path = os.path.join(base_dir, "data", "cifar-10-batches-py")
local_hf_path = os.path.join(base_dir, "data", "cifar10_hf")
local_torchvision_root = "data"

def get_resnet_cnn_model(num_classes: int = 10) -> nn.Module:
    return tiny_resnet18(num_classes=num_classes, base_width=8)

def get_basic_cnn_model() -> nn.Module:
    return Net()

def load_data(partition_id: int, num_partitions: int, alpha_val: float, backdoor_enabled: bool = False,
              target_label: int = 2, poison_fraction: float = 0.1):

    global fds
    if fds is None:
        if not os.path.isdir(local_hf_path):
            raise RuntimeError(
                f"Offline mode: expected HF dataset at {local_hf_path}. "
                "Run your pre-download step and copy it here."
            )

        hf_ds = load_from_disk(local_hf_path)
        hf_train = hf_ds["train"]

        global dirichlet_cache
        if dirichlet_cache is None:
            labels = hf_train["label"]
            dirichlet_cache = dirichlet_indices(
                labels=labels,
                num_partitions=num_partitions,
                alpha=alpha_val,
                seed=42,
            )

        fds = []
        for indices in dirichlet_cache:
            fds.append(hf_train.select(indices))

    partition = fds[partition_id]
    partition_train_test = partition.train_test_split(test_size=0.2, seed=42)

    pytorch_transforms = v2.Compose([
        v2.ToImage(),
        v2.RandomCrop(32, padding=4),
        v2.RandomHorizontalFlip(),
        v2.ColorJitter(0.2, 0.2, 0.2, 0.1),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize((0.4914, 0.4822, 0.4465),
                     (0.2023, 0.1994, 0.2010))
    ])

    pytorch_test_transforms = Compose([
        ToTensor(),
        Normalize((0.4914, 0.4822, 0.4465),
                  (0.2023, 0.1994, 0.2010)),
    ])

    def apply_train_transforms(batch):
        batch["img"] = [pytorch_transforms(img) for img in batch["img"]]
        return batch

    def apply_test_transforms(batch):
        batch["img"] = [pytorch_test_transforms(img) for img in batch["img"]]
        return batch

    partition_train = partition_train_test["train"].with_transform(apply_train_transforms)
    partition_backdoor_train = partition_train_test["train"].with_transform(apply_test_transforms)
    partition_test = partition_train_test["test"].with_transform(apply_test_transforms)

    cuda_avail = torch.cuda.is_available()
    num_workers = 0 if os.name == "nt" else 2
    pin_memory = True if cuda_avail else False

    if backdoor_enabled:
        training_data = DataLoader(
            partition_train,
            batch_size=64,
            shuffle=True,
            collate_fn=lambda batch: collate_with_backdoor(batch, num_backdoor_per_batch=20, target_label=target_label),
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
    else:
        training_data = DataLoader(
            partition_train,
            batch_size=64,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

    test_data = DataLoader(partition_test, batch_size=64, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)

    return training_data, test_data

def train(net, training_data, epochs, device, lr=0.1):
    """Train the model on the training set using SGD + CosineAnnealingLR and label smoothing."""
    net.to(device)
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.05).to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))

    net.train()
    running_loss = 0.0
    for _ in range(epochs):
        for batch in training_data:
            if isinstance(batch, dict):
                images, labels = batch["img"], batch["label"]
            else:
                images, labels = batch
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)

            optimizer.zero_grad()
            outputs = net(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        scheduler.step()

    avg_training_loss = running_loss / len(training_data)
    final_vec = parameters_to_vector(net.parameters()).detach().cpu().clone()
    return avg_training_loss, final_vec

def train_backdoor(net, training_data, epochs, device, lr=0.01):
    """Train the model on the training set using SGD + CosineAnnealingLR and label smoothing."""
    net.to(device)
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=1e-6)

    net.train()
    running_loss = 0.0
    for _ in range(epochs):
        for batch in training_data:
            if isinstance(batch, dict):
                images, labels = batch["img"], batch["label"]
            else:
                images, labels = batch
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)

            optimizer.zero_grad()
            outputs = net(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

    avg_training_loss = running_loss / len(training_data)
    final_vec = parameters_to_vector(net.parameters()).detach().cpu().clone()
    return avg_training_loss, final_vec

def test(net, test_data, device):
    net.to(device)
    net.eval()
    criterion = torch.nn.CrossEntropyLoss()
    correct, total, loss = 0, 0, 0.0
    with torch.no_grad():
        for batch in test_data:
            if isinstance(batch, dict):
                images, labels = batch["img"], batch["label"]
            else:
                images, labels = batch
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return loss / len(test_data), correct / total

def get_weights(net):
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


def set_weights(net, parameters):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)

def load_test_data_for_eval(batch_size=64):
    """Load CIFAR-10 test data offline (prefers local torchvision files, then HF copy)."""

    pytorch_transforms = Compose(
        [ToTensor(), Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]
    )

    """
    local_torch_path = os.path.join("data", "cifar-10-batches-py")
    if os.path.isdir(local_torch_path):
        test_dataset = CIFAR10(
            root="data",
            train=False,
            download=False,       # never download
            transform=pytorch_transforms,
        )
        return DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    """

    if os.path.isdir(local_hf_path):
        from datasets import load_from_disk
        hf_ds = load_from_disk(local_hf_path)

        def apply_transforms(batch):
            batch["img"] = [pytorch_transforms(img) for img in batch["img"]]
            return batch

        hf_ds = hf_ds.with_transform(apply_transforms)
        return DataLoader(hf_ds["test"], batch_size=batch_size, shuffle=False)

    raise RuntimeError(
        "Offline mode: CIFAR-10 dataset not found. "
        "Expected either './data/cifar-10-batches-py/' (torchvision) "
        "or './data/cifar10_hf/' (Hugging Face)."
    )


def test_eval(net, test_data, device):
    """Evaluate the updated model on the test set for evaluations."""
    net.to(device)
    net.eval()
    criterion = torch.nn.CrossEntropyLoss()
    correct, loss = 0, 0.0
    with torch.no_grad():
        for batch in test_data:
            if isinstance(batch, dict):
                images, labels = batch["img"], batch["label"]
            else:
                images, labels = batch
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
    accuracy = correct / len(test_data.dataset)
    loss = loss / len(test_data)
    return loss, accuracy
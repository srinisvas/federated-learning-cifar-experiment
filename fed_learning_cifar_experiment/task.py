"""fed-learning-cifar-experiment: A Flower / PyTorch app."""

from collections import OrderedDict
from pathlib import Path

import torch
import torch.nn as nn
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import DirichletPartitioner
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.transforms import Compose, Normalize, ToTensor

from fed_learning_cifar_experiment.utils.backdoor_attack import collate_with_backdoor
from fed_learning_cifar_experiment.models.basic_cnn_model import Net
from fed_learning_cifar_experiment.models.resnet_cnn_model import tiny_resnet18

fds = None  # Cache FederatedDataset
DATA_ROOT = Path("./data")

def get_resnet_cnn_model(num_classes: int = 10) -> nn.Module:
    return tiny_resnet18(num_classes=num_classes, base_width=8)

def get_basic_cnn_model() -> nn.Module:
    return Net()

def load_data(partition_id: int, num_partitions: int, alpha_val: float, backdoor_enabled: bool = False,
              target_label: int = 2, poison_fraction: float = 0.1):
    """Load partition CIFAR10 data."""
    global fds
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    if fds is None:
        #Using Dirichlet Partitioner - with alpha - 0.9
        partitioner = DirichletPartitioner(num_partitions=num_partitions, alpha=alpha_val, partition_by="label")
        fds = FederatedDataset(
            dataset="uoft-cs/cifar10",
            partitioners={"train": partitioner},
            cache_dir=str(DATA_ROOT / "federated_cache"),
        )
    partition = fds.load_partition(partition_id)
    # Divide data on each node: 80% train, 20% test
    partition_train_test = partition.train_test_split(test_size=0.2, seed=42)
    pytorch_transforms = Compose(
        [ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    def apply_transforms(batch):
        """Apply transforms to the partition from FederatedDataset."""
        batch["img"] = [pytorch_transforms(img) for img in batch["img"]]
        return batch

    partition_train_test = partition_train_test.with_transform(apply_transforms)

    """Logic to poison a percentage of images
    if backdoor_enabled:
        num_poison = int(len(partition_train_test) * poison_fraction)
        poisoned_indices = torch.randperm(len(partition_train_test))[:num_poison]
        for idx in poisoned_indices:
            img, _ = partition_train_test[idx]
            partition_train_test[idx] = (add_trigger(img), target_label)
    """

    if backdoor_enabled:
        training_data = DataLoader(
            partition_train_test["train"],
            batch_size=64,
            shuffle=True,
            collate_fn=lambda batch: collate_with_backdoor(batch, num_backdoor_per_batch=20, target_label=2)
        )
        test_data = DataLoader(partition_train_test["test"], batch_size=64)
    else:
        training_data = DataLoader(partition_train_test["train"], batch_size=64, shuffle=True)
        test_data = DataLoader(partition_train_test["test"], batch_size=64)

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
            images = batch["img"]
            labels = batch["label"]
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
    net.eval()
    criterion = torch.nn.CrossEntropyLoss()
    correct, loss = 0, 0.0
    with torch.no_grad():
        for batch in test_data:
            images = batch["img"].to(device)
            labels = batch["label"].to(device)
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

    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    test_dataset = CIFAR10(root=str(DATA_ROOT), train=False, download=True, transform=pytorch_transforms)

    test_data = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return test_data

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
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
    accuracy = correct / len(test_data.dataset)
    loss = loss / len(test_data)
    return loss, accuracy
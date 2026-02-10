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
import torch.nn.functional as F
from torch.nn.utils import parameters_to_vector, vector_to_parameters

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
        v2.ColorJitter(0.1, 0.1, 0.1, 0.05),
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
    num_workers = 0 #if os.name == "nt" else 2
    pin_memory = False #True if cuda_avail else False

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

def train(net, training_data, epochs, device, lr=0.05):
    """Train the model on the training set using SGD + CosineAnnealingLR and label smoothing."""
    net.to(device)
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.05).to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=len(training_data) * epochs
    )

    net.train()
    running_loss = 0.0
    for epoch in range(epochs):
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
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=0)
    #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=len(training_data) * epochs
    )

    net.train()
    running_loss = 0.0
    for epoch in range(epochs):
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

def train_constrain_and_scale(
    net,
    training_data,
    epochs,
    device,
    init_vec: torch.Tensor,                 # g_t
    prev_global_vec: torch.Tensor = None,   # g_{t-1} (recommended)
    lr: float = 0.005,

    # --- Camouflage weights (good defaults) ---
    lambda_norm: float = 0.02,              # keep update small-ish (distance camouflage)
    lambda_dir: float = 0.50,               # align with global drift direction
    lambda_target_norm: float = 0.10,       # match typical update norm (helps MultiKrum normalize_updates=True)
    lambda_pair: float = 0.20,

    # --- Norm target control ---
    target_delta_norm: float = None,        # if None, estimate from ||g_t - g_{t-1}||
    min_dir_norm: float = 1e-12,

    # --- Early stop (optional) ---
    epsilon_ce: float = None,               # keep None until stable

    # --- Loss ---
    label_smoothing: float = 0.0,           # keep 0 for attacker usually
):
    """
    Constrain-and-scale optimized for Krum / MultiKrum.

    Shapes the attacker UPDATE delta = w - g_t:
      - small L2 (distance camouflage)
      - direction aligned with d_t = g_t - g_{t-1}
      - norm matching to a target magnitude (useful when MultiKrum normalizes updates)

    Returns: final_vec (weights) on CPU
    """

    net.to(device)
    net.train()

    # Start exactly from g_t
    g = init_vec.detach().to(device)
    vector_to_parameters(g, net.parameters())

    # Direction d_t (unit) from last global drift, if available
    d_unit = None
    g_prev = None

    if prev_global_vec is not None:
        g_prev = prev_global_vec.detach().to(device)
        d = (g - g_prev)
        d_norm = torch.norm(d)
        if d_norm >= min_dir_norm:
            d_unit = d / d_norm

        # If no explicit target norm, estimate from last global step magnitude
        if target_delta_norm is None:
            est = float(torch.norm(g - g_prev).detach().cpu())
            if est >= 1e-8:
                target_delta_norm = est

    criterion = torch.nn.CrossEntropyLoss(label_smoothing=label_smoothing).to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=0.0)

    for epoch in range(epochs):
        running_ce = 0.0
        steps = 0

        for batch in training_data:
            if isinstance(batch, dict):
                images, labels = batch["img"], batch["label"]
            else:
                images, labels = batch

            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            logits = net(images)
            ce = criterion(logits, labels)

            w = parameters_to_vector(net.parameters())
            delta = (w - g)

            # (1) keep update small -> helps Krum distances
            l_norm = torch.mean(delta * delta)

            loss = ce + lambda_norm * l_norm

            # (2) direction camouflage -> helps look like benign drift
            if d_unit is not None and lambda_dir > 0.0:
                delta_norm = torch.norm(delta) + 1e-12
                delta_unit = delta / delta_norm
                cos = torch.dot(delta_unit, d_unit).clamp(-1.0, 1.0)
                l_dir = 1.0 - cos
                loss = loss + lambda_dir * l_dir

            # (3) norm matching -> helps MultiKrum normalize_updates=True
            if target_delta_norm is not None and lambda_target_norm > 0.0:
                delta_norm = torch.norm(delta) + 1e-12
                l_tnorm = (delta_norm - target_delta_norm) ** 2
                loss = loss + lambda_target_norm * l_tnorm

            # (4) pairwise camouflage (Krum proxy): match benign-like update vector
            if prev_global_vec is not None and lambda_pair > 0.0:
                # benign proxy update from last global step
                delta_ref = (g - g_prev)
                l_pair = torch.mean((delta - delta_ref) ** 2)
                loss = loss + lambda_pair * l_pair

            # --- Warm-up: let CE dominate early ---
            camouflage_scale = 1.0
            if prev_global_vec is not None:
                # Ramp camouflage over epochs
                warmup_epochs = max(1, epochs // 3)
                camouflage_scale = min(1.0, epoch / max(1, epochs // 3))

            # Base task objective
            loss = ce

            # (1) norm camouflage
            loss += camouflage_scale * lambda_norm * l_norm

            # (2) direction camouflage
            if d_unit is not None and lambda_dir > 0.0:
                loss += camouflage_scale * lambda_dir * l_dir

            # (3) target-norm matching
            if target_delta_norm is not None and lambda_target_norm > 0.0:
                loss += camouflage_scale * lambda_target_norm * l_tnorm

            # (4) pairwise Krum-proxy camouflage
            if g_prev is not None and lambda_pair > 0.0:
                loss += camouflage_scale * lambda_pair * l_pair

            loss.backward()
            optimizer.step()

            running_ce += float(ce.detach().cpu())
            steps += 1

        if epsilon_ce is not None and (running_ce / max(1, steps)) < epsilon_ce:
            break

    return parameters_to_vector(net.parameters()).detach().cpu().clone()


def krum_safe_scale(
    final_vec: torch.Tensor,
    init_vec: torch.Tensor,
    gamma: float,
    keep_delta_norm: bool = False,
):
    """
    Krum-safe scaling: scaled_vec = g + gamma*(w - g)

    For Krum: gamma must be small (typically 1 to 5).
    For MultiKrum with normalize_updates=True: scaling changes direction if later normalized,
    but still keep gamma small to avoid outlier distances pre-normalization.

    If keep_delta_norm=True, we rescale back to the original delta norm after applying gamma.
    Useful if you want to preserve "typical magnitude" while nudging direction.
    """
    g = init_vec.detach().cpu()
    w = final_vec.detach().cpu()
    delta = w - g

    scaled_delta = gamma * delta

    if keep_delta_norm:
        orig = torch.norm(delta) + 1e-12
        new = torch.norm(scaled_delta) + 1e-12
        scaled_delta = scaled_delta * (orig / new)

    return (g + scaled_delta).clone()


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

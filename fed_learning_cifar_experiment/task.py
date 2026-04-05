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
from torch.nn.utils import parameters_to_vector, vector_to_parameters
import copy
import torch
import torch.nn.functional as F
import numpy as np
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

@torch.no_grad()
def _clone_net(net):
    # Safe deep copy for PyTorch modules
    return copy.deepcopy(net)


def build_reference_clean_deltas(
    net,
    training_data,
    device,
    init_vec: torch.Tensor,
    epochs: int = 1,
    lr: float = 0.05,
    num_refs: int = 6,
    seed_base: int = 1234,
    label_smoothing: float = 0.05,
):
    """
    Build a small set of "honest-like" reference deltas using ONLY local clean training.
    Each reference delta is produced by training from init_vec with a different seed.
    Returns: list[torch.Tensor] of deltas on CPU, each shaped [D]
    """
    refs = []
    init_vec_cpu = init_vec.detach().cpu()

    for r in range(num_refs):
        torch.manual_seed(seed_base + r)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed_base + r)

        net_r = _clone_net(net)
        net_r.to(device)
        net_r.train()

        # Start from global (init_vec)
        g = init_vec_cpu.to(device)
        vector_to_parameters(g, net_r.parameters())

        criterion = torch.nn.CrossEntropyLoss(label_smoothing=label_smoothing).to(device)
        optimizer = torch.optim.SGD(net_r.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)

        # Simple schedule is OK, keep it consistent with your benign client settings
        for _ in range(epochs):
            for batch in training_data:
                if isinstance(batch, dict):
                    images, labels = batch["img"], batch["label"]
                else:
                    images, labels = batch
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)
                logits = net_r(images)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()

        w_r = parameters_to_vector(net_r.parameters()).detach().cpu()
        delta_r = (w_r - init_vec_cpu).clone()
        refs.append(delta_r)

    return refs

def krum_score_proxy(delta, ref_deltas, k):
    diffs = ref_deltas - delta.unsqueeze(0)
    dists = torch.sum(diffs * diffs, dim=1)
    return torch.topk(dists, k=min(k, len(dists)), largest=False).values.mean()

def train_constrain_and_scale_krum_proxy(
    net,
    training_data,
    device,
    init_vec,
    clean_delta,
    ref_clean_deltas=None,
    krum_ref_delta=None,

    # Optim
    epochs=3,                     # reduced
    lr=0.01,
    label_smoothing=0.0,
    weight_decay=0.0,

    # Stealth weights
    lambda_norm_match=0.1,
    lambda_krum_proxy=0.25,       # slightly stronger geometry
    lambda_centroid=0.0,
    lambda_anchor=0.05,
    lambda_temporal=0.0,          #  removed

    prev_malicious_delta=None,

    krum_k=7,
    ref_scale=1.0,
    eps=1e-12,
):
    import torch
    import torch.nn.functional as F
    from torch.nn.utils import parameters_to_vector, vector_to_parameters

    net = net.to(device)
    net.train()

    g = init_vec.detach().to(device)
    vector_to_parameters(g, net.parameters())

    if isinstance(ref_clean_deltas, torch.Tensor):
        refs = ref_clean_deltas.to(device)
    else:
        refs = torch.stack(ref_clean_deltas, dim=0).to(device)

    clean_delta_dev = clean_delta.to(device)
    clean_norm = torch.norm(clean_delta_dev) + eps

    # SHARED DENSE ANCHOR (your improved version)
    diffs = refs.unsqueeze(1) - refs.unsqueeze(0)
    pairwise = torch.sum(diffs * diffs, dim=2)

    k_eff = min(krum_k, refs.shape[0] - 1)
    knn = torch.topk(pairwise, k=k_eff, largest=False).values
    scores = torch.mean(knn, dim=1)

    best_id = torch.argmin(scores)
    anchor = refs[best_id].detach()

    top2 = torch.topk(-scores, k=min(2, len(scores))).indices
    if len(top2) >= 2:
        anchor = (0.85 * refs[top2[0]] + 0.15 * refs[top2[1]]).detach()

    criterion = torch.nn.CrossEntropyLoss(label_smoothing=label_smoothing).to(device)

    # -------------------------
    # Stage 1: Backdoor ONLY
    # -------------------------
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9)

    for _ in range(1):   #only 1 epoch
        for batch in training_data:
            images, labels = batch if not isinstance(batch, dict) else (batch["img"], batch["label"])
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = net(images)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
            optimizer.step()

    # -------------------------
    # Stage 2: Geometry shaping (short)
    # -------------------------
    optimizer = torch.optim.SGD(net.parameters(), lr=lr * 0.5, momentum=0.9)

    for _ in range(1):
        for i, batch in enumerate(training_data):

            images, labels = batch if not isinstance(batch, dict) else (batch["img"], batch["label"])
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad(set_to_none=True)

            logits = net(images)
            ce = criterion(logits, labels)

            w = parameters_to_vector(net.parameters())
            delta_adv = w - g
            adv_norm = torch.norm(delta_adv) + eps

            # Norm
            norm_match = (adv_norm - clean_norm) ** 2

            #Krum proxy (lighter)
            diff = refs - delta_adv.unsqueeze(0)
            dists = torch.sum(diff * diff, dim=1)
            k = min(krum_k, dists.numel())
            knn_vals = torch.topk(dists, k=k, largest=False).values
            knn_loss = torch.mean(knn_vals[: max(1, k // 2)])

            # Anchor
            anchor_loss = torch.mean((delta_adv - anchor) ** 2)

            loss = (
                0.4 * ce
                + lambda_norm_match * norm_match
                + lambda_krum_proxy * knn_loss
                + lambda_anchor * anchor_loss
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
            optimizer.step()

        # -------------------------
        # Final Projection (Version V2 for cold starts)
        # -------------------------

        with torch.no_grad():
            w = parameters_to_vector(net.parameters())
            delta_adv = w - g

            ref_norms = torch.norm(refs, dim=1)
            ref_centroid = torch.mean(refs, dim=0)

            # Find the tightest cluster in ref space (lowest variance neighborhood)
            diffs = refs.unsqueeze(1) - refs.unsqueeze(0)
            pairwise = torch.sum(diffs * diffs, dim=2)
            k_eff = min(krum_k, refs.shape[0] - 1)
            knn_scores = torch.topk(pairwise, k=k_eff, largest=False).values.mean(dim=1)
            best_ref = refs[torch.argmin(knn_scores)]  # Krum-optimal ref delta

            # Pull toward Krum-optimal ref rather than plain centroid
            dist_to_best = torch.norm(delta_adv - best_ref)
            ref_spread = torch.std(
                torch.norm(refs - ref_centroid.unsqueeze(0), dim=1)
            ) + eps

            if dist_to_best > ref_spread:
                pull = (dist_to_best - ref_spread) / (dist_to_best + eps)
                pull = torch.clamp(pull, 0.0, 0.75)
                delta_adv = (1.0 - pull) * delta_adv + pull * best_ref

            # Norm: stay within benign range with floor
            ref_median_norm = torch.median(ref_norms)
            target_norm = torch.max(ref_median_norm, 0.5 * clean_norm)
            current_norm = torch.norm(delta_adv) + eps
            delta_adv = delta_adv * (target_norm / current_norm)

            vector_to_parameters(g + delta_adv, net.parameters())

    return parameters_to_vector(net.parameters()).detach().cpu().clone()

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
                alpha=0.9,
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


def train_constrain_and_scale_paper(
    net,
    training_data,
    device,
    init_vec: torch.Tensor,               # G_t (flattened global model)

    # --- Algorithm 1 hyperparameters ---
    epochs: int = 10,
    lr: float = 0.01,
    alpha: float = 0.5,                    # balance: alpha*L_class + (1-alpha)*L_ano
    gamma: float = None,                   # post-train scale factor; if None, use n/eta
    n_participants: int = 100,             # total participants n
    eta: float = 1.0,                      # global learning rate eta
    gamma_bound: float = None,             # max allowed gamma (anomaly detector bound S)

    # --- Anomaly loss type ---
    ano_type: str = "l2",                  # "l2" | "cosine" | "l2+cosine" | "krum" | "krum+l2"
    lambda_cosine: float = 1.0,            # weight for cosine term when using "l2+cosine"

    # --- Krum-aware L_ano (required when ano_type contains "krum") ---
    ref_deltas: torch.Tensor = None,       # [num_refs, D] reference benign deltas
    krum_k: int = 7,                       # k-nearest neighbors for Krum proxy
    lambda_l2: float = 0.01,              # L2 weight in "krum+l2" mode

    # --- LR schedule ---
    step_schedule: list = None,            # epochs at which to decay LR
    step_rate: float = 10.0,               # LR divisor at each step

    # --- Early stop ---
    epsilon_ce: float = None,              # stop if L_class < epsilon

    # --- Loss config ---
    label_smoothing: float = 0.0,

    # --- Post-training scaling mode ---
    scale_mode: str = "gamma",             # "gamma" | "norm_match"
):
    """
    Algorithm 1 from Bagdasaryan et al. (2020) with defense-aware L_ano.

    The paper's framework is: L = alpha * L_class + (1-alpha) * L_ano
    where L_ano is pluggable for the target defense.

    ano_type options:
      "l2"       - ||X - G_t||^2. For FedAvg / weight-magnitude detectors.
      "cosine"   - 1 - cos(delta, G_t). For cosine-similarity detectors.
      "l2+cosine"- weighted combination.
      "krum"     - Krum proxy: minimize sum of k-nearest distances to
                   reference benign deltas. For Krum / MultiKrum.
      "krum+l2"  - Krum proxy + mild L2 norm regularization. Recommended
                   for Krum since it optimizes cluster proximity AND
                   keeps update magnitude reasonable.

    scale_mode options:
      "gamma"      - standard post-train scaling by gamma (paper default).
      "norm_match" - scale delta to match median benign delta norm.
                     Stealthier for Krum since it avoids magnitude outliers.
    """
    net = net.to(device)
    net.train()

    # Validate Krum mode requirements
    uses_krum = "krum" in ano_type
    if uses_krum and ref_deltas is None:
        raise ValueError(
            f"ano_type='{ano_type}' requires ref_deltas (benign reference "
            "deltas from the server). Pass shared_ref_deltas from config."
        )

    # Step 1: Initialize X = G_t
    g = init_vec.detach().to(device)
    vector_to_parameters(g, net.parameters())

    # Move references to device if needed
    refs_dev = None
    ref_median_norm = None
    if ref_deltas is not None:
        if isinstance(ref_deltas, np.ndarray):
            ref_deltas = torch.from_numpy(ref_deltas).float()
        refs_dev = ref_deltas.to(device)
        ref_median_norm = torch.median(torch.norm(refs_dev, dim=1)).item()

    # Default LR schedule
    if step_schedule is None:
        step_schedule = [epochs // 2, 3 * epochs // 4]

    criterion = torch.nn.CrossEntropyLoss(label_smoothing=label_smoothing).to(device)
    current_lr = lr
    optimizer = torch.optim.SGD(net.parameters(), lr=current_lr, momentum=0.9)

    eps = 1e-12

    # Step 2-3: Train with unified loss
    for epoch in range(epochs):

        # LR step decay
        if epoch in step_schedule:
            current_lr /= step_rate
            for pg in optimizer.param_groups:
                pg["lr"] = current_lr

        running_class_loss = 0.0
        steps = 0

        for batch in training_data:
            if isinstance(batch, dict):
                images, labels = batch["img"], batch["label"]
            else:
                images, labels = batch
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            # L_class: CE on mixed batch (clean + backdoor via collate_with_backdoor)
            logits = net(images)
            l_class = criterion(logits, labels)

            # L_ano: defense-specific anomaly penalty
            w = parameters_to_vector(net.parameters())
            delta = w - g

            if ano_type == "l2":
                l_ano = torch.sum(delta * delta)

            elif ano_type == "cosine":
                cos_sim = F.cosine_similarity(delta.unsqueeze(0), g.unsqueeze(0))
                l_ano = 1.0 - cos_sim.squeeze()

            elif ano_type == "l2+cosine":
                l2_term = torch.sum(delta * delta)
                cos_sim = F.cosine_similarity(delta.unsqueeze(0), g.unsqueeze(0))
                cos_term = 1.0 - cos_sim.squeeze()
                l_ano = l2_term + lambda_cosine * cos_term

            elif ano_type == "krum":
                # Krum proxy: minimize distance to k-nearest reference deltas
                # This directly optimizes what Krum measures for selection
                diff = refs_dev - delta.unsqueeze(0)          # [num_refs, D]
                dists_sq = torch.sum(diff * diff, dim=1)      # [num_refs]
                k_eff = min(krum_k, dists_sq.numel())
                knn_vals = torch.topk(dists_sq, k=k_eff, largest=False).values
                l_ano = torch.mean(knn_vals)

            elif ano_type == "krum+l2":
                # Krum proximity + mild L2 to keep norm reasonable
                diff = refs_dev - delta.unsqueeze(0)
                dists_sq = torch.sum(diff * diff, dim=1)
                k_eff = min(krum_k, dists_sq.numel())
                knn_vals = torch.topk(dists_sq, k=k_eff, largest=False).values
                krum_loss = torch.mean(knn_vals)
                l2_loss = torch.sum(delta * delta)
                l_ano = krum_loss + lambda_l2 * l2_loss

            else:
                raise ValueError(f"Unknown ano_type: {ano_type}")

            # Unified loss (Eq. 4 from the paper)
            loss = alpha * l_class + (1.0 - alpha) * l_ano

            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
            optimizer.step()

            running_class_loss += float(l_class.detach().cpu())
            steps += 1

        avg_class_loss = running_class_loss / max(1, steps)
        if epsilon_ce is not None and avg_class_loss < epsilon_ce:
            break

    # Step 4: Post-training scaling
    with torch.no_grad():
        x = parameters_to_vector(net.parameters()).detach()
        delta_final = x - g

        if scale_mode == "norm_match" and ref_median_norm is not None:
            # Scale delta to match median benign update norm
            # This is the stealthiest option for Krum: magnitude looks benign
            current_norm = torch.norm(delta_final).item()
            if current_norm > eps:
                scale = ref_median_norm / current_norm
                scaled_vec = g + scale * delta_final
            else:
                scaled_vec = x
        else:
            # Standard gamma scaling (paper default)
            if gamma is None:
                gamma = n_participants / eta

            if gamma_bound is not None:
                delta_norm = torch.norm(delta_final).item()
                if delta_norm > eps:
                    max_gamma = gamma_bound / delta_norm
                    gamma = min(gamma, max_gamma)

            scaled_vec = g + gamma * delta_final

        vector_to_parameters(scaled_vec, net.parameters())

    final_vec = parameters_to_vector(net.parameters()).detach().cpu().clone()
    return avg_class_loss, final_vec


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

            # --- Warm-up: let CE dominate early ---
            camouflage_scale = 1.0
            if g_prev is not None:
                warmup_epochs = max(1, epochs // 3)
                camouflage_scale = min(1.0, epoch / warmup_epochs)

            # Base task objective
            loss = ce

            # (1) norm camouflage
            loss += camouflage_scale * lambda_norm * torch.mean(delta * delta)

            # (2) direction camouflage
            if d_unit is not None and lambda_dir > 0.0:
                delta_norm = torch.norm(delta) + 1e-12
                delta_norm = torch.sqrt(torch.sum(delta * delta) + 1e-12)
                delta_unit = delta / delta_norm
                cos = torch.dot(delta_unit, d_unit).clamp(-1.0, 1.0)
                loss += camouflage_scale * lambda_dir * (1.0 - cos)

            # (3) target-norm matching
            if target_delta_norm is not None and lambda_target_norm > 0.0:
                delta_norm = torch.norm(delta) + 1e-12
                delta_norm = torch.sqrt(torch.sum(delta * delta) + 1e-12)
                loss += camouflage_scale * lambda_target_norm * (
                        delta_norm - target_delta_norm
                ) ** 2

            # (4) pairwise Krum-proxy camouflage
            if g_prev is not None and lambda_pair > 0.0:
                delta_ref = (g - g_prev)
                loss += camouflage_scale * lambda_pair * torch.mean(
                    (delta - delta_ref) ** 2
                )

            # (5) CRITICAL: prevent delta collapsing to zero
            if target_delta_norm is not None:
                min_attack_norm = 0.05 * target_delta_norm  # tune 0.02–0.10
                min_sq = (min_attack_norm ** 2)

                delta_sq = torch.sum(delta * delta)  # ||delta||^2
                loss += F.relu(min_sq - delta_sq) ** 2

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
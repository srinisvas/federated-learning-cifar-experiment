"""
Per-update structural feature extraction for FL Byzantine-robust defense
research.

Extracts ~66 features per (round, client) pair from the structure of model
updates alone — no client self-reported signals, no server-side validation
data. Designed to enable:
  (a) the three-population separation analysis
      (near-IID benign | extreme non-IID benign | malicious),
  (b) downstream cross-client / temporal modeling,
  (c) post-hoc analysis without re-running experiments.

Design choices worth knowing:
  - Operates on `net.parameters()` only. BN running stats (running_mean,
    running_var, num_batches_tracked) are excluded because they accumulate
    via data exposure, not gradient steps; their "delta" magnitude depends
    on BatchNorm momentum and number of batches seen, not on the update
    signal we want to characterize.
  - Stages = {stem, layer1, layer2, layer3, layer4, head} for TinyResNet18.
    Six stages, matching the doc.
  - Spectral features: per-tensor SVD on each 2D+ weight, then
    Frobenius-norm-weighted average across tensors in the stage. Avoids the
    "pad with zeros and SVD" pathology.
  - Cross-stage coordination: the literal `cosine(stage_l, stage_{l+1})` from
    the doc is ill-posed because consecutive stages have different parameter
    counts in ResNet (e.g., stem ~240, layer4 ~131k). Replaced with a single
    well-defined feature: cosine of the per-stage-L2-norm signature against
    the round-mean signature. Captures "is this client's energy distribution
    across stages similar to the population's?" — which is what the original
    feature was reaching for.
"""

import os
import math
import numpy as np
import torch
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple, Any, Iterable

EPS = 1e-12


# ---------------------------------------------------------------------------
# Stage definitions for TinyResNet18.
# ---------------------------------------------------------------------------

STAGE_PREDICATES: List[Tuple[str, Any]] = [
    ("stem",   lambda k: k.startswith("conv1.") or k.startswith("bn1.")),
    ("layer1", lambda k: k.startswith("layer1.")),
    ("layer2", lambda k: k.startswith("layer2.")),
    ("layer3", lambda k: k.startswith("layer3.")),
    ("layer4", lambda k: k.startswith("layer4.")),
    ("head",   lambda k: k.startswith("fc.")),
]
STAGE_NAMES: List[str] = [name for name, _ in STAGE_PREDICATES]


def _is_trainable_key(k: str) -> bool:
    """Exclude BN running statistics and num_batches_tracked."""
    return not (
        k.endswith("running_mean")
        or k.endswith("running_var")
        or k.endswith("num_batches_tracked")
    )


def _stage_of(k: str) -> Optional[str]:
    for name, pred in STAGE_PREDICATES:
        if pred(k):
            return name
    return None


# ---------------------------------------------------------------------------
# Param registry
#
# Built once from a model instance. Maps each trainable param name to its
# stage, shape, flat-vector offset, and numel. Shared across all extraction
# calls in a run so we don't repeatedly re-introspect the model.
# ---------------------------------------------------------------------------

class ParamRegistry:
    """One-time introspection of the model's trainable parameter layout."""

    def __init__(self, model: torch.nn.Module):
        self.entries: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
        self.state_dict_keys: List[str] = list(model.state_dict().keys())
        offset = 0
        for k, v in model.named_parameters():
            stage = _stage_of(k)
            if stage is None:
                # Defensive: should never happen for TinyResNet18.
                raise ValueError(
                    f"Unmatched param '{k}'. Update STAGE_PREDICATES."
                )
            self.entries[k] = {
                "stage": stage,
                "shape": tuple(v.shape),
                "offset": offset,
                "numel": v.numel(),
                "dim": v.dim(),
            }
            offset += v.numel()
        self.total_trainable_params: int = offset
        self.stage_to_keys: "OrderedDict[str, List[str]]" = OrderedDict(
            (s, []) for s in STAGE_NAMES
        )
        for k, e in self.entries.items():
            self.stage_to_keys[e["stage"]].append(k)


# ---------------------------------------------------------------------------
# Conversion helpers
# ---------------------------------------------------------------------------

def nds_to_trainable_state_dict(
    nds: List[np.ndarray],
    state_dict_keys: List[str],
) -> Dict[str, torch.Tensor]:
    """
    Flower's nds list is ordered like state_dict().items(). Build a dict
    containing only trainable params (BN running stats dropped).
    """
    out: Dict[str, torch.Tensor] = {}
    for k, arr in zip(state_dict_keys, nds):
        if not _is_trainable_key(k):
            continue
        out[k] = torch.from_numpy(np.asarray(arr)).float()
    return out


def state_dict_delta(
    client_sd: Dict[str, torch.Tensor],
    global_sd: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """delta[k] = client[k] - global[k], over the intersection of keys."""
    return {k: client_sd[k] - global_sd[k] for k in client_sd.keys()}


def flat_param_vec_to_per_key_dict(
    flat: torch.Tensor,
    registry: ParamRegistry,
) -> Dict[str, torch.Tensor]:
    """
    Slice a flat vector laid out by net.parameters() ordering (e.g., your
    ref_deltas) back into a dict keyed by param name with original shapes.
    """
    flat = flat.float()
    out: Dict[str, torch.Tensor] = {}
    for k, e in registry.entries.items():
        seg = flat[e["offset"]: e["offset"] + e["numel"]]
        out[k] = seg.reshape(e["shape"])
    return out


def mean_delta_per_key(
    deltas: Iterable[Dict[str, torch.Tensor]],
) -> Dict[str, torch.Tensor]:
    """Element-wise mean across a list of per-key delta dicts."""
    deltas = list(deltas)
    if not deltas:
        return {}
    keys = deltas[0].keys()
    return {k: torch.stack([d[k] for d in deltas], dim=0).mean(dim=0) for k in keys}


# ---------------------------------------------------------------------------
# Scalar feature primitives
# ---------------------------------------------------------------------------

def _kurtosis(x: torch.Tensor) -> float:
    """Excess kurtosis (Fisher's). NaN if numerically degenerate."""
    if x.numel() < 4:
        return float("nan")
    x = x.float()
    mu = x.mean()
    var = x.var(unbiased=False)
    if float(var) < EPS:
        return float("nan")
    z = (x - mu) / torch.sqrt(var + EPS)
    return float((z ** 4).mean().item() - 3.0)


def _skewness(x: torch.Tensor) -> float:
    if x.numel() < 3:
        return float("nan")
    x = x.float()
    mu = x.mean()
    var = x.var(unbiased=False)
    if float(var) < EPS:
        return float("nan")
    z = (x - mu) / torch.sqrt(var + EPS)
    return float((z ** 3).mean().item())


def _cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    if a.numel() == 0 or b.numel() == 0:
        return float("nan")
    a = a.float()
    b = b.float()
    den = float(torch.norm(a) * torch.norm(b)) + EPS
    if den < EPS:
        return float("nan")
    return float(torch.dot(a, b).item() / den)


def _per_tensor_spectral(weight: torch.Tensor) -> Optional[Tuple[float, float, float]]:
    """
    For a 2D+ tensor: SVD on the [out_ch, fan_in_flat] reshape.
    Returns (top_sv_ratio, spectral_entropy, frobenius_norm).
    None for 1D tensors (BN gamma/beta, biases).
    """
    if weight.dim() < 2 or weight.numel() == 0:
        return None
    m = weight.reshape(weight.shape[0], -1).float()
    if m.shape[0] == 0 or m.shape[1] == 0:
        return None
    try:
        s = torch.linalg.svdvals(m)
    except Exception:
        return None
    s_sum = float(s.sum().item()) + EPS
    top_sv_ratio = float(s[0].item() / s_sum)
    p = (s / (s.sum() + EPS)).clamp(min=EPS)
    spec_entropy = float(-(p * torch.log(p)).sum().item())
    fro = float(torch.norm(m).item())
    return top_sv_ratio, spec_entropy, fro


def _stage_spectral_aggregate(
    delta_per_key: Dict[str, torch.Tensor],
    keys_in_stage: List[str],
) -> Tuple[float, float]:
    """
    Frobenius-weighted average of (top_sv_ratio, spectral_entropy) across all
    2D+ tensors in a stage. NaN if no 2D+ tensors (shouldn't happen for any
    TinyResNet18 stage but handled defensively).
    """
    weights, top_ratios, spec_entropies = [], [], []
    for k in keys_in_stage:
        out = _per_tensor_spectral(delta_per_key[k])
        if out is None:
            continue
        top_sv_ratio, spec_entropy, fro = out
        weights.append(fro)
        top_ratios.append(top_sv_ratio)
        spec_entropies.append(spec_entropy)
    if not weights:
        return float("nan"), float("nan")
    w = np.asarray(weights, dtype=np.float64)
    w_norm = w / (w.sum() + EPS)
    return (
        float(np.sum(w_norm * np.asarray(top_ratios))),
        float(np.sum(w_norm * np.asarray(spec_entropies))),
    )


def _per_class_delta_entropy(fc_weight_delta: torch.Tensor) -> float:
    """
    fc_weight_delta: [num_classes, feat_dim].
    Shannon entropy over per-class L2 magnitudes — the privacy-respecting
    proxy for client-side class diversity.
    Max value = log(num_classes); for CIFAR-10, log(10) ~= 2.302.
    Low entropy => update concentrated on few output classes (suspicious or
    extreme non-IID); high entropy => spread across many classes.
    """
    if fc_weight_delta.dim() != 2 or fc_weight_delta.shape[0] < 2:
        return float("nan")
    per_class_mag = torch.norm(fc_weight_delta.float(), dim=1)
    s = float(per_class_mag.sum().item()) + EPS
    p = (per_class_mag / s).clamp(min=EPS)
    return float(-(p * torch.log(p)).sum().item())


# ---------------------------------------------------------------------------
# Top-level extraction
# ---------------------------------------------------------------------------

def extract_per_update_features(
    *,
    client_delta_per_key: Dict[str, torch.Tensor],
    round_mean_delta_per_key: Dict[str, torch.Tensor],
    registry: ParamRegistry,
    reference_delta_per_key: Optional[Dict[str, torch.Tensor]] = None,
) -> "OrderedDict[str, float]":
    """
    Compute the full feature dict for ONE client's update.

    Args:
      client_delta_per_key:    delta tensors for this client, keyed by param name.
      round_mean_delta_per_key: element-wise mean of all sampled clients' deltas
                                 this round (computed over the trainable param space).
      registry:                ParamRegistry built once from the model.
      reference_delta_per_key: optional clean-reference delta (mean of ref_deltas);
                                 enables Block 5 reference-comparison features.

    Returns: OrderedDict of feature_name -> float. Always returns the same set
    of keys (NaN for anything that couldn't be computed), so the CSV schema is
    stable across calls.
    """
    out: "OrderedDict[str, float]" = OrderedDict()

    keys_ordered = list(registry.entries.keys())

    # ---- whole-update geometry ----
    full_client = torch.cat([client_delta_per_key[k].flatten() for k in keys_ordered])
    full_round_mean = torch.cat([round_mean_delta_per_key[k].flatten() for k in keys_ordered])

    out["total_update_l2_norm"] = float(torch.norm(full_client).item())
    out["cosine_to_round_mean_update"] = _cosine(full_client, full_round_mean)
    out["l2_distance_to_round_mean"] = float(torch.norm(full_client - full_round_mean).item())

    if reference_delta_per_key is not None:
        full_ref = torch.cat([reference_delta_per_key[k].flatten() for k in keys_ordered])
        out["cosine_to_reference_clean_delta"] = _cosine(full_client, full_ref)
        out["l2_distance_to_reference"] = float(torch.norm(full_client - full_ref).item())
    else:
        out["cosine_to_reference_clean_delta"] = float("nan")
        out["l2_distance_to_reference"] = float("nan")

    # ---- per-stage features ----
    stage_l2_norms: Dict[str, float] = {}
    for stage in STAGE_NAMES:
        keys = registry.stage_to_keys[stage]
        if keys:
            x = torch.cat([client_delta_per_key[k].flatten() for k in keys])
        else:
            x = torch.zeros(0)

        # Magnitude
        l2 = float(torch.norm(x).item())
        stage_l2_norms[stage] = l2
        linf = float(x.abs().max().item()) if x.numel() else 0.0
        out[f"stage_{stage}_l2_norm"] = l2
        out[f"stage_{stage}_linf_to_l2_ratio"] = (
            (linf / (l2 + EPS)) if l2 > EPS else float("nan")
        )

        # Distributional shape
        out[f"stage_{stage}_kurtosis"] = _kurtosis(x)
        out[f"stage_{stage}_skewness"] = _skewness(x)

        # Spectral (per-tensor SVD, Frobenius-weighted aggregate within stage)
        per_key = {k: client_delta_per_key[k] for k in keys}
        top_sv_ratio, spec_entropy = _stage_spectral_aggregate(per_key, keys)
        out[f"stage_{stage}_top_sv_ratio"] = top_sv_ratio
        out[f"stage_{stage}_spectral_entropy"] = spec_entropy

        # Directional: cosine to round-mean stage delta
        if keys:
            stage_round_mean = torch.cat([round_mean_delta_per_key[k].flatten() for k in keys])
        else:
            stage_round_mean = torch.zeros(0)
        out[f"stage_{stage}_cos_to_round_mean"] = _cosine(x, stage_round_mean)

        # Reference comparison (per-stage)
        if reference_delta_per_key is not None and keys:
            stage_ref = torch.cat([reference_delta_per_key[k].flatten() for k in keys])
            out[f"stage_{stage}_cos_to_reference"] = _cosine(x, stage_ref)
        else:
            out[f"stage_{stage}_cos_to_reference"] = float("nan")

    # Energy distribution: ratio of each stage's L2 to total L2
    total_l2 = out["total_update_l2_norm"] + EPS
    for stage in STAGE_NAMES:
        out[f"stage_{stage}_norm_ratio_to_total"] = stage_l2_norms[stage] / total_l2

    # ---- cross-stage structural features ----
    # Stage-norm signature: a 6-vector summarizing magnitude per stage.
    # Compare to the round-mean signature => one well-defined "energy
    # distribution coordination" scalar. This replaces the doc's literal
    # cosine(Δ_l, Δ_{l+1}) which is dimension-mismatched in ResNet.
    sig_client = torch.tensor([stage_l2_norms[s] for s in STAGE_NAMES], dtype=torch.float32)
    sig_round_mean = torch.tensor(
        [
            float(
                torch.norm(
                    torch.cat([round_mean_delta_per_key[k].flatten() for k in registry.stage_to_keys[s]])
                    if registry.stage_to_keys[s] else torch.zeros(0)
                ).item()
            )
            for s in STAGE_NAMES
        ],
        dtype=torch.float32,
    )
    out["stage_norm_signature_cos_to_round_mean"] = _cosine(sig_client, sig_round_mean)

    # Classifier-to-backbone norm ratio
    head_l2 = stage_l2_norms.get("head", 0.0)
    backbone_l2 = sum(stage_l2_norms[s] for s in STAGE_NAMES if s != "head")
    out["classifier_to_backbone_norm_ratio"] = head_l2 / (backbone_l2 + EPS)

    # Per-class delta entropy: from fc.weight delta (privacy-preserving
    # class-coverage proxy)
    if "fc.weight" in client_delta_per_key:
        out["per_class_delta_entropy"] = _per_class_delta_entropy(
            client_delta_per_key["fc.weight"]
        )
    else:
        out["per_class_delta_entropy"] = float("nan")

    # Top-2 stage energy concentration
    sorted_l2 = sorted(stage_l2_norms.values(), reverse=True)
    out["stage_energy_concentration_top2"] = (
        sum(sorted_l2[:2]) / (sum(sorted_l2) + EPS)
    )

    return out


# ---------------------------------------------------------------------------
# Convenience: return the canonical feature key list (useful for asserting
# CSV schema stability or for downstream loaders).
# ---------------------------------------------------------------------------

def canonical_feature_keys() -> List[str]:
    """Return the list of all feature keys produced by extract_per_update_features,
    in emission order. Use to assert CSV schema stability or for downstream loaders."""
    keys = [
        "total_update_l2_norm",
        "cosine_to_round_mean_update",
        "l2_distance_to_round_mean",
        "cosine_to_reference_clean_delta",
        "l2_distance_to_reference",
    ]
    # Per-stage block emitted inside the stage loop (8 features per stage).
    per_stage_features = [
        "l2_norm",
        "linf_to_l2_ratio",
        "kurtosis",
        "skewness",
        "top_sv_ratio",
        "spectral_entropy",
        "cos_to_round_mean",
        "cos_to_reference",
    ]
    for stage in STAGE_NAMES:
        for feat in per_stage_features:
            keys.append(f"stage_{stage}_{feat}")
    # norm_ratio_to_total emitted after the stage loop (needs total L2 first)
    for stage in STAGE_NAMES:
        keys.append(f"stage_{stage}_norm_ratio_to_total")
    # Cross-stage / global structural features
    keys.extend([
        "stage_norm_signature_cos_to_round_mean",
        "classifier_to_backbone_norm_ratio",
        "per_class_delta_entropy",
        "stage_energy_concentration_top2",
    ])
    return keys
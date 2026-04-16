"""
Per-update structural feature extraction for FL Byzantine-robust defense
research.

Extracts structural features per (round, client) pair from the structure of model
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

Features are grouped by the "adversarial squeeze" they target — pairs of
features that are inversely related through the structure of the backdoor
gradient, so gaming one forces the other to drift. See design doc for the
five squeezes.

Additional squeeze-targeting features (beyond the original 63):
  - head_backbone_conservation_ratio: operationalizes signal conservation
    (suppressing head signal surfaces as backbone kurtosis)
  - backbone_kurtosis_max: companion to the above
  - cos_to_mean_leave_one_out (whole + per-stage): unbiased version of
    cos_to_round_mean, harder to game (attacker faces 10 different targets)
  - stage_pair_projected_cos (5 features): cross-layer gradient-flow
    consistency via projection onto per-stage honest SVD basis. Captures
    dual-objective leakage (Principle 2) directly.
  - head_sign_agreement_with_ref: fraction of head delta params sharing
    sign with reference — catches backdoor concentration on target-class row
    even when cosine/magnitude are matched.
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


def build_leave_one_out_means(
    client_deltas: List[Dict[str, torch.Tensor]],
) -> List[Dict[str, torch.Tensor]]:
    """
    Build per-client leave-one-out means. Output[i] = mean over all deltas
    except client i. O(n) in memory per key: compute total sum once, then
    subtract each client's contribution.
    """
    n = len(client_deltas)
    if n < 2:
        return [{k: v.clone() for k, v in d.items()} for d in client_deltas]
    keys = client_deltas[0].keys()
    # Sum across all clients per key
    totals = {k: torch.stack([d[k] for d in client_deltas], dim=0).sum(dim=0)
              for k in keys}
    out = []
    for i in range(n):
        out.append({k: (totals[k] - client_deltas[i][k]) / (n - 1) for k in keys})
    return out


def build_stage_projection_bases(
    ref_deltas_per_key: List[Dict[str, torch.Tensor]],
    registry: "ParamRegistry",
    k: int = 3,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """
    For each stage, fit a k-dim SVD basis from the reference deltas' deltas
    in that stage. Also project the reference mean onto each basis to get
    the "typical" position in projected coords.

    Returns:
      bases:         {stage: [k_eff, D_stage] tensor} (k_eff <= k if fewer refs)
      ref_proj_means: {stage: [k_eff] tensor} — mean of refs in projected coords

    Called once per round, from the strategy, using the same ref_deltas that
    are already sent to clients.
    """
    bases: Dict[str, torch.Tensor] = {}
    ref_proj_means: Dict[str, torch.Tensor] = {}
    if not ref_deltas_per_key:
        return bases, ref_proj_means

    for stage in STAGE_NAMES:
        stage_keys = registry.stage_to_keys[stage]
        if not stage_keys:
            continue
        basis = _fit_stage_projection_basis(ref_deltas_per_key, stage_keys, k=k)
        if basis is None:
            continue
        bases[stage] = basis
        # Project each ref's stage delta, then average
        proj_refs = []
        for ref in ref_deltas_per_key:
            p = _project_stage_delta(ref, stage_keys, basis)
            if p is not None:
                proj_refs.append(p)
        if proj_refs:
            ref_proj_means[stage] = torch.stack(proj_refs, dim=0).mean(dim=0)
    return bases, ref_proj_means


def ref_deltas_flat_to_per_key_list(
    ref_deltas_flat: np.ndarray,
    registry: "ParamRegistry",
) -> List[Dict[str, torch.Tensor]]:
    """
    Convert the server-side ref_deltas array [num_refs, D_params] (flat,
    net.parameters() ordering) into a list of per-key delta dicts, so it
    can be consumed by build_stage_projection_bases.
    """
    out = []
    for i in range(ref_deltas_flat.shape[0]):
        flat = torch.from_numpy(ref_deltas_flat[i]).float()
        out.append(flat_param_vec_to_per_key_dict(flat, registry))
    return out


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


def _leave_one_out_mean(
    all_client_stacked: torch.Tensor,
    client_idx: int,
) -> torch.Tensor:
    """
    Given a stacked tensor of shape [n_clients, D] and a target index,
    return the mean over all OTHER clients (excludes this client).
    This is the unbiased version of round_mean — removes the self-bias
    that inflates cos_to_round_mean toward 1.

    Adversarial property: the attacker faces n different leave-one-out
    means (one per position), so matching all of them simultaneously is
    structurally harder than matching a single shared round mean.
    """
    n = all_client_stacked.shape[0]
    if n < 2:
        return all_client_stacked[0].clone()
    total = all_client_stacked.sum(dim=0)
    return (total - all_client_stacked[client_idx]) / (n - 1)


def _head_sign_agreement_with_ref(
    fc_weight_delta: torch.Tensor,
    ref_fc_weight_delta: torch.Tensor,
) -> float:
    """
    Fraction of fc.weight delta parameters sharing sign with the reference
    delta. Honest clients ~ 0.5 (data-driven randomness around the mean
    direction). Attackers with backdoor concentration show systematic sign
    bias on the target-class row because the backdoor gradient has a fixed
    sign structure (push trigger-responsive features toward the target
    class's logit).

    Computed on the flattened fc weight delta. The effect is strongest on
    the target-class row but doesn't require knowing which row is targeted:
    any systematic sign bias across the whole head is suspicious.
    """
    if fc_weight_delta.numel() == 0 or ref_fc_weight_delta.numel() == 0:
        return float("nan")
    if fc_weight_delta.shape != ref_fc_weight_delta.shape:
        return float("nan")
    # Elements where both have meaningful sign (ignore near-zero)
    a = fc_weight_delta.flatten()
    b = ref_fc_weight_delta.flatten()
    mask = (a.abs() > EPS) & (b.abs() > EPS)
    if mask.sum().item() == 0:
        return float("nan")
    agree = ((a[mask] > 0) == (b[mask] > 0)).float().mean()
    return float(agree.item())


def _fit_stage_projection_basis(
    ref_deltas_per_key: List[Dict[str, torch.Tensor]],
    stage_keys: List[str],
    k: int = 3,
) -> Optional[torch.Tensor]:
    """
    Fit a low-dim projection basis for one stage from N reference deltas.

    Stacks all refs' deltas for this stage into a [N, D_stage] matrix, runs
    SVD, returns the top-k right singular vectors (basis of shape [k, D_stage]).
    With N=6 ref_deltas and k=3, each ref contributes ~2 effective directions.

    Returns None if the stage has too few params or SVD fails.

    The basis changes each round (recomputed from this round's ref_deltas).
    This is intentional: the global model shifts each round, and the honest
    delta distribution shifts with it. A cached basis would measure against
    an increasingly stale reference.
    """
    if not ref_deltas_per_key or not stage_keys:
        return None
    stage_vecs = []
    for ref in ref_deltas_per_key:
        try:
            v = torch.cat([ref[k].flatten() for k in stage_keys]).float()
        except KeyError:
            return None
        stage_vecs.append(v)
    M = torch.stack(stage_vecs, dim=0)  # [N, D_stage]
    if M.shape[1] == 0:
        return None
    try:
        # Center to get directions of variation, not mean direction
        M_centered = M - M.mean(dim=0, keepdim=True)
        # SVD: M = U S V^T, rows of V^T are right singular vectors
        _, _, Vt = torch.linalg.svd(M_centered, full_matrices=False)
    except Exception:
        return None
    k_eff = min(k, Vt.shape[0])
    if k_eff == 0:
        return None
    return Vt[:k_eff].detach()  # [k, D_stage]


def _project_stage_delta(
    delta_per_key: Dict[str, torch.Tensor],
    stage_keys: List[str],
    basis: torch.Tensor,
) -> Optional[torch.Tensor]:
    """
    Project a client's stage delta onto the k-dim basis fit from refs.
    Returns a k-dimensional vector, or None on shape/key errors.
    """
    if basis is None or not stage_keys:
        return None
    try:
        v = torch.cat([delta_per_key[k].flatten() for k in stage_keys]).float()
    except KeyError:
        return None
    if v.numel() != basis.shape[1]:
        return None
    return basis @ v  # [k]


# ---------------------------------------------------------------------------
# Top-level extraction
# ---------------------------------------------------------------------------

def extract_per_update_features(
    *,
    client_delta_per_key: Dict[str, torch.Tensor],
    round_mean_delta_per_key: Dict[str, torch.Tensor],
    registry: ParamRegistry,
    reference_delta_per_key: Optional[Dict[str, torch.Tensor]] = None,
    # ---- New optional arguments (for squeeze-targeting features) ----
    leave_one_out_mean_per_key: Optional[Dict[str, torch.Tensor]] = None,
    stage_projection_bases: Optional[Dict[str, torch.Tensor]] = None,
    stage_projection_ref_means: Optional[Dict[str, torch.Tensor]] = None,
) -> "OrderedDict[str, float]":
    """
    Compute the full feature dict for ONE client's update.

    Args:
      client_delta_per_key:       delta tensors for this client, keyed by param name.
      round_mean_delta_per_key:   element-wise mean of all sampled clients' deltas
                                   this round.
      registry:                   ParamRegistry built once from the model.
      reference_delta_per_key:    optional clean-reference delta (mean of ref_deltas);
                                   enables reference-comparison features.
      leave_one_out_mean_per_key: optional LOO mean for THIS client (excludes self);
                                   enables cos_to_mean_leave_one_out features.
      stage_projection_bases:     optional dict {stage: [k, D_stage] basis tensor}
                                   fit from per-round ref_deltas; enables
                                   stage_pair_projected_cos features (Principle 2).
      stage_projection_ref_means: optional dict {stage: [k] ref-mean vector in
                                   projected coords}; companion to bases.

    Returns: OrderedDict of feature_name -> float. Schema is stable across
    calls — NaN is filled for anything that couldn't be computed.
    """
    out: "OrderedDict[str, float]" = OrderedDict()

    keys_ordered = list(registry.entries.keys())

    # ---- whole-update geometry ----
    full_client = torch.cat([client_delta_per_key[k].flatten() for k in keys_ordered])
    full_round_mean = torch.cat([round_mean_delta_per_key[k].flatten() for k in keys_ordered])

    out["total_update_l2_norm"] = float(torch.norm(full_client).item())
    out["cosine_to_round_mean_update"] = _cosine(full_client, full_round_mean)
    out["l2_distance_to_round_mean"] = float(torch.norm(full_client - full_round_mean).item())

    # Leave-one-out cosine: unbiased version (attacker faces n different targets)
    if leave_one_out_mean_per_key is not None:
        full_loo = torch.cat([leave_one_out_mean_per_key[k].flatten() for k in keys_ordered])
        out["cosine_to_mean_leave_one_out"] = _cosine(full_client, full_loo)
    else:
        out["cosine_to_mean_leave_one_out"] = float("nan")

    if reference_delta_per_key is not None:
        full_ref = torch.cat([reference_delta_per_key[k].flatten() for k in keys_ordered])
        out["cosine_to_reference_clean_delta"] = _cosine(full_client, full_ref)
        out["l2_distance_to_reference"] = float(torch.norm(full_client - full_ref).item())
    else:
        out["cosine_to_reference_clean_delta"] = float("nan")
        out["l2_distance_to_reference"] = float("nan")

    # ---- per-stage features ----
    stage_l2_norms: Dict[str, float] = {}
    stage_kurtoses: Dict[str, float] = {}
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
        kurt = _kurtosis(x)
        stage_kurtoses[stage] = kurt if not math.isnan(kurt) else 0.0
        out[f"stage_{stage}_kurtosis"] = kurt
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

        # Per-stage leave-one-out cosine
        if leave_one_out_mean_per_key is not None and keys:
            stage_loo = torch.cat([leave_one_out_mean_per_key[k].flatten() for k in keys])
            out[f"stage_{stage}_cos_to_mean_loo"] = _cosine(x, stage_loo)
        else:
            out[f"stage_{stage}_cos_to_mean_loo"] = float("nan")

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

    # Per-class delta entropy
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

    # ---- Squeeze-targeting features ----

    # Squeeze 2: head-backbone signal conservation
    # Max kurtosis across backbone stages (excludes stem and head).
    # Rationale: the backdoor signal surfaces somewhere in the backbone when
    # the attacker suppresses head signal. Max across layer1..layer4 catches
    # whichever stage the attacker used to encode the hidden signal.
    backbone_stages = ["layer1", "layer2", "layer3", "layer4"]
    backbone_kurts = [stage_kurtoses[s] for s in backbone_stages]
    out["backbone_kurtosis_max"] = float(max(backbone_kurts))
    # The conservation feature: product of head energy and backbone kurtosis.
    # Honest clients have moderate head ratio AND moderate backbone kurtosis
    # => moderate product. Attackers are pushed toward (low head, high kurt)
    # or (high head, low kurt) — the product is useful because it becomes
    # elevated only when BOTH are elevated, which is the signature of an
    # attacker trying to hide head signal in backbone spike patterns.
    out["head_backbone_conservation_product"] = (
        out["stage_head_norm_ratio_to_total"] * out["backbone_kurtosis_max"]
    )

    # Head target-row sign agreement with reference
    # Squeeze 5 companion: sign structure of fc.weight delta vs reference
    if reference_delta_per_key is not None and "fc.weight" in client_delta_per_key:
        out["head_sign_agreement_with_ref"] = _head_sign_agreement_with_ref(
            client_delta_per_key["fc.weight"],
            reference_delta_per_key["fc.weight"],
        )
    else:
        out["head_sign_agreement_with_ref"] = float("nan")

    # Principle 2: cross-layer gradient-flow consistency via projected cosines.
    # For each consecutive stage pair, project both stages' deltas onto their
    # respective per-round ref-delta SVD bases, then cosine-compare against
    # the ref mean's projected position. Captures whether the client's
    # cross-layer structure follows the natural gradient-flow correlations
    # that emerge in honest training.
    stage_pairs = list(zip(STAGE_NAMES[:-1], STAGE_NAMES[1:]))  # 5 pairs
    for s_from, s_to in stage_pairs:
        feat_name = f"stage_pair_{s_from}_to_{s_to}_projected_cos"
        if stage_projection_bases is None or stage_projection_ref_means is None:
            out[feat_name] = float("nan")
            continue
        basis_from = stage_projection_bases.get(s_from)
        basis_to = stage_projection_bases.get(s_to)
        ref_proj_from = stage_projection_ref_means.get(s_from)
        ref_proj_to = stage_projection_ref_means.get(s_to)
        if basis_from is None or basis_to is None or ref_proj_from is None or ref_proj_to is None:
            out[feat_name] = float("nan")
            continue
        # Project this client's stage deltas
        proj_from = _project_stage_delta(
            client_delta_per_key, registry.stage_to_keys[s_from], basis_from
        )
        proj_to = _project_stage_delta(
            client_delta_per_key, registry.stage_to_keys[s_to], basis_to
        )
        if proj_from is None or proj_to is None:
            out[feat_name] = float("nan")
            continue
        # Relative-to-ref coords: subtract ref mean, then cosine between the
        # (from - ref_from) and (to - ref_to) deviations. Honest clients'
        # deviations are jointly determined by their data (chain rule ties
        # adjacent stages). Attackers' per-stage camouflage doesn't preserve
        # this joint structure.
        dev_from = proj_from - ref_proj_from
        dev_to = proj_to - ref_proj_to
        out[feat_name] = _cosine(dev_from, dev_to)

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
        "cosine_to_mean_leave_one_out",
        "cosine_to_reference_clean_delta",
        "l2_distance_to_reference",
    ]
    # Per-stage block emitted inside the stage loop (9 features per stage now).
    per_stage_features = [
        "l2_norm",
        "linf_to_l2_ratio",
        "kurtosis",
        "skewness",
        "top_sv_ratio",
        "spectral_entropy",
        "cos_to_round_mean",
        "cos_to_mean_loo",
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
    # Squeeze-targeting features
    keys.extend([
        "backbone_kurtosis_max",
        "head_backbone_conservation_product",
        "head_sign_agreement_with_ref",
    ])
    # Stage-pair projected cosines (5 pairs for 6 stages)
    stage_pairs = list(zip(STAGE_NAMES[:-1], STAGE_NAMES[1:]))
    for s_from, s_to in stage_pairs:
        keys.append(f"stage_pair_{s_from}_to_{s_to}_projected_cos")
    return keys
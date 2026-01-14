import os
import json
import random
import logging
from typing import Any, Dict, List, Optional, Tuple

import torch
import flwr as fl
from flwr.common import FitIns
from flwr.common import parameters_to_ndarrays

from fed_learning_cifar_experiment.utils.logger import (
    append_distributed_round,
    write_experiment_summary,
)


# -------------------- helpers --------------------

def _flatten(nds, *, device: torch.device) -> torch.Tensor:
    return torch.cat(
        [torch.as_tensor(a, dtype=torch.float32, device=device).reshape(-1) for a in nds]
    )

def _mad(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    med = x.median()
    return (x - med).abs().median() + eps

def _robust_z_median(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return 0.6745 * (x - x.median()) / _mad(x, eps)

def _robust_z_centered(
    x: torch.Tensor, center: torch.Tensor, scale: torch.Tensor, eps: float = 1e-12
) -> torch.Tensor:
    return 0.6745 * (x - center) / (scale + eps)

def _cos_dist(u: torch.Tensor, v: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    u = u / (u.norm() + eps)
    v = v / (v.norm() + eps)
    return 1.0 - torch.clamp(torch.dot(u, v), -1.0, 1.0)

def _forward_cluster(unit_dirs: torch.Tensor, theta_s: float) -> List[List[int]]:
    if len(unit_dirs) == 0:
        return []

    clusters: List[List[int]] = []
    current = [0]
    centroid = unit_dirs[0].clone()

    for i in range(1, len(unit_dirs)):
        if float(_cos_dist(centroid, unit_dirs[i])) < theta_s:
            current.append(i)
            centroid = centroid * (len(current) - 1) + unit_dirs[i]
            centroid = centroid / (centroid.norm() + 1e-12)
        else:
            clusters.append(current)
            current = [i]
            centroid = unit_dirs[i].clone()

    clusters.append(current)
    return clusters

def _cluster_centroid(unit_dirs: torch.Tensor, idxs: List[int]) -> torch.Tensor:
    c = unit_dirs[idxs].mean(dim=0)
    return c / (c.norm() + 1e-12)


# -------------------- combined strategy --------------------

class SaveFedAvgMetricsClusterDefenseStrategy(fl.server.strategy.FedAvg):
    """
    Combines:
      1) Your custom configure_fit() (malicious sampling + per-client config)
      2) FedAvg + clustering defense in aggregate_fit()
      3) Your aggregate_evaluate() metrics persistence (MTA/ASR)
      4) record_centralized_eval()

    No per-client history is stored as part of defense.
    """

    def __init__(
        self,
        simulation_id: str = "",
        num_clients: int = 0,
        num_rounds: int = 0,
        aggregation_method: str = "",
        backdoor_attack_mode: str = "",
        num_of_malicious_clients: int = 0,
        num_of_malicious_clients_per_round: int = 0,

        # defense knobs
        theta_s: float = 0.05,
        z_thresh: float = 3.5,
        min_cluster_size: int = 3,
        use_angle_signal: bool = True,
        max_drop_fraction: float = 0.33,
        persist_round_stats: bool = True,

        device: Optional[torch.device] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        # ---- your existing metadata ----
        self.simulation_id = simulation_id
        self.num_clients = num_clients
        self.num_rounds = num_rounds
        self.aggregation_method = aggregation_method
        self.backdoor_attack_mode = backdoor_attack_mode

        self.history = {"round": [], "mta": [], "asr": []}
        self.central_mta_history = []
        self.central_asr_history = []
        self.final_centralized_mta = None
        self.final_centralized_asr = None

        self.num_of_malicious_clients = num_of_malicious_clients
        self.num_of_malicious_clients_per_round = num_of_malicious_clients_per_round

        # ---- defense config ----
        self.theta_s = float(theta_s)
        self.z_thresh = float(z_thresh)
        self.min_cluster_size = int(min_cluster_size)
        self.use_angle_signal = bool(use_angle_signal)
        self.max_drop_fraction = float(max_drop_fraction)
        self.persist_round_stats = bool(persist_round_stats)

        self.device = device if device is not None else torch.device("cpu")

        self.round_state: Dict[str, Any] = {
            "median_norm": None,
            "mad_norm": None,
        }

        self.round_state_path = f"runs/{simulation_id}/round_aggregates.jsonl"

        self.logger = logging.getLogger(self.__class__.__name__)

        # Keep a global reference model for delta computation
        # NOTE: Strategy has initial_parameters in FedAvg __init__ only if passed.
        # We will lazily set it when first round happens.
        self._global_nd = None

    # ---------------- customization hooks (unchanged behavior) ----------------

    def configure_fit(self, server_round: int, parameters, client_manager):
        num_available = len(client_manager.all())
        sample_size, min_num = self.num_fit_clients(num_available)
        sampled_clients = list(client_manager.sample(sample_size, min_num))
        sampled_ids = [c.cid for c in sampled_clients]

        # Randomly pick malicious clients from the sampled list
        num_malicious = min(self.num_of_malicious_clients_per_round, len(sampled_ids))
        malicious_ids = random.sample(sampled_ids, num_malicious)

        fit_ins_list = []
        for client in sampled_clients:
            config = self.on_fit_config_fn(server_round) if self.on_fit_config_fn else {}
            config.update({
                "current-round": server_round,
                "sampled_client_ids": json.dumps(sampled_ids),
                "malicious_client_ids": json.dumps(malicious_ids),
                "is_malicious": str(client.cid in malicious_ids),
            })
            fit_ins_list.append((client, FitIns(parameters, config)))

        return fit_ins_list

    def record_centralized_eval(self, rnd, loss, mta, asr):
        self.central_mta_history.append(mta)
        self.central_asr_history.append(asr)
        if rnd == self.num_rounds:
            self.final_centralized_mta = mta
            self.final_centralized_asr = asr

    # ---------------- defense happens here ----------------

    def aggregate_fit(self, server_round, results, failures):
        if not results:
            return None, {}

        # Initialize global reference lazily
        if self._global_nd is None:
            # Use first incoming client's params as initial global reference
            self._global_nd = parameters_to_ndarrays(results[0][1].parameters)

        w_global = _flatten(self._global_nd, device=self.device)

        # Build deltas + norms
        deltas: List[torch.Tensor] = []
        norms_list: List[torch.Tensor] = []
        entries: List[Tuple[Any, Any]] = []

        for client, fit_res in results:
            nds = parameters_to_ndarrays(fit_res.parameters)
            delta = _flatten(nds, device=self.device) - w_global
            deltas.append(delta)
            norms_list.append(delta.norm())
            entries.append((client, fit_res))

        deltas_t = torch.stack(deltas)
        norms = torch.stack(norms_list).to(self.device)
        unit_dirs = deltas_t / (norms.unsqueeze(1) + 1e-12)

        # Sort by projection on population mean direction
        pop_mean_dir = unit_dirs.mean(dim=0)
        pop_mean_dir = pop_mean_dir / (pop_mean_dir.norm() + 1e-12)
        proj = torch.matmul(unit_dirs, pop_mean_dir)
        order = torch.argsort(proj, descending=True)

        unit_dirs = unit_dirs[order]
        norms = norms[order]
        entries = [entries[i] for i in order.tolist()]

        # Cluster (greedy sweep)
        clusters = _forward_cluster(unit_dirs, theta_s=self.theta_s)
        cluster_sizes = [len(c) for c in clusters]

        # Intra-round detection
        local_flag = torch.zeros(len(norms), dtype=torch.bool, device=self.device)
        global_norm_flag = _robust_z_median(norms) > self.z_thresh

        for idxs in clusters:
            if len(idxs) >= self.min_cluster_size:
                # norm anomaly
                z_norm = _robust_z_median(norms[idxs])
                flag = z_norm > self.z_thresh

                # angular anomaly (optional)
                if self.use_angle_signal:
                    centroid = _cluster_centroid(unit_dirs, idxs)
                    ang = 1.0 - torch.clamp(unit_dirs[idxs] @ centroid, -1.0, 1.0)
                    z_ang = _robust_z_median(ang)
                    flag = flag | (z_ang > self.z_thresh)

                local_flag[idxs] = flag
            else:
                # fallback so defense doesn't switch off in small clusters
                local_flag[idxs] = global_norm_flag[idxs]

        # Inter-round population check (confidence boost)
        hist_flag = torch.zeros(len(norms), dtype=torch.bool, device=self.device)
        if self.round_state["median_norm"] is not None and self.round_state["mad_norm"] is not None:
            prev_med = torch.tensor(self.round_state["median_norm"], device=self.device)
            prev_mad = torch.tensor(self.round_state["mad_norm"], device=self.device)
            z_prev = _robust_z_centered(norms, center=prev_med, scale=prev_mad).abs()
            hist_flag = z_prev > self.z_thresh

        high_conf = local_flag & hist_flag
        low_conf = local_flag | hist_flag

        # default: drop only high-confidence ones
        final_flag = high_conf.clone()

        # optional conservative expansion: drop low_conf if it won't remove too many
        max_drop = max(1, int(len(results) * self.max_drop_fraction))
        if int(final_flag.sum()) == 0 and int(low_conf.sum()) <= max_drop:
            final_flag = low_conf

        filtered = [e for e, bad in zip(entries, final_flag.tolist()) if not bad]

        # hard fallback: if we removed too many, revert
        if len(filtered) < max(2, len(results) // 2):
            filtered = entries
            final_flag[:] = False

        print(
            f"[Round {server_round}] "
            f"clusters={len(clusters)} sizes={cluster_sizes} "
            f"local={int(local_flag.sum())} hist={int(hist_flag.sum())} "
            f"final_drop={int(final_flag.sum())}"
        )

        # Run FedAvg aggregation on filtered set
        params, metrics = super().aggregate_fit(server_round, filtered, failures)

        # Update global reference
        if params is not None:
            self._global_nd = parameters_to_ndarrays(params)

        # Update inter-round statistics
        self.round_state["median_norm"] = float(norms.median().item())
        self.round_state["mad_norm"] = float(_mad(norms).item())

        # Persist stats if requested
        if self.persist_round_stats:
            os.makedirs(os.path.dirname(self.round_state_path), exist_ok=True)
            with open(self.round_state_path, "a", encoding="utf-8") as f:
                f.write(json.dumps({
                    "round": server_round,
                    "num_clients": len(results),
                    "num_clusters": len(clusters),
                    "cluster_sizes": cluster_sizes,
                    "flag_local": int(local_flag.sum().item()),
                    "flag_hist": int(hist_flag.sum().item()),
                    "flag_high_conf": int(high_conf.sum().item()),
                    "flag_final": int(final_flag.sum().item()),
                    "median_norm": self.round_state["median_norm"],
                    "mad_norm": self.round_state["mad_norm"],
                    "theta_s": self.theta_s,
                    "z_thresh": self.z_thresh,
                    "min_cluster_size": self.min_cluster_size,
                    "use_angle_signal": self.use_angle_signal,
                }) + "\n")

        return params, metrics

    # ---------------- evaluation logging (your original) ----------------

    def aggregate_evaluate(self, rnd, results, failures):
        metrics = super().aggregate_evaluate(rnd, results, failures)

        mta_vals = [res.metrics.get("mta", 0.0) for _, res in results]
        asr_vals = [res.metrics.get("asr", 0.0) for _, res in results]

        avg_mta = sum(mta_vals) / len(mta_vals) if mta_vals else 0.0
        avg_asr = sum(asr_vals) / len(asr_vals) if asr_vals else 0.0

        self.history["round"].append(rnd)
        self.history["mta"].append(avg_mta)
        self.history["asr"].append(avg_asr)

        print(f"[Round {rnd}] MTA={avg_mta:.4f}, ASR={avg_asr:.4f}")

        dist_loss = metrics[0] if metrics else None
        append_distributed_round(
            self.simulation_id,
            rnd,
            avg_mta,
            avg_asr,
            dist_loss,
            self.num_clients,
        )

        # final summary
        if rnd >= self.num_rounds:
            dist_mta = self.history.get("mta", [])
            dist_asr = self.history.get("asr", [])

            write_experiment_summary(
                simulation_id=self.simulation_id,
                meta={
                    "aggregation": str(self.aggregation_method),
                    "num_rounds": str(self.num_rounds),
                    "num_malicious_clients": str(self.num_of_malicious_clients),
                    "backdoor_attack_mode": str(self.backdoor_attack_mode),
                    "alpha": 0.9,
                },
                final_centralized_mta=self.final_centralized_mta or 0.0,
                final_centralized_asr=self.final_centralized_asr or 0.0,
                dist_mta_history=dist_mta,
                dist_asr_history=dist_asr,
                central_mta_history=self.central_mta_history,
                central_asr_history=self.central_asr_history,
                notes=""
            )

        return metrics

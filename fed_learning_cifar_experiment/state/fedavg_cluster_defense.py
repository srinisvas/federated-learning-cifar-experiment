import os
import json
from typing import Any, Dict, List

import torch
import flwr as fl
from flwr.common import parameters_to_ndarrays


# -------------------- helpers --------------------

def _flatten(nds):
    return torch.cat([torch.tensor(a, dtype=torch.float32).flatten() for a in nds])

def _mad(x, eps=1e-12):
    med = x.median()
    return (x - med).abs().median() + eps

def _robust_z(x, eps=1e-12):
    return 0.6745 * (x - x.median()) / _mad(x, eps)

def _cos_dist(u, v):
    return 1.0 - torch.dot(u, v).clamp(-1.0, 1.0)

def _forward_cluster(unit_dirs, theta_s):
    clusters = []
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


# -------------------- strategy --------------------

class FedAvgClusterDefenseStrategy(fl.server.strategy.FedAvg):
    """
    FedAvg + intra-round directional clustering + inter-round aggregate validation.
    No per-client history is stored.
    """

    def __init__(
        self,
        simulation_id: str,
        initial_parameters,
        persist_round_stats: bool = True,
        **kwargs,
    ):
        super().__init__(initial_parameters=initial_parameters, **kwargs)

        self.simulation_id = simulation_id
        self.persist_round_stats = persist_round_stats

        self._global_nd = parameters_to_ndarrays(initial_parameters)

        # population-level inter-round state
        self.round_state: Dict[str, Any] = {
            "mean_norm": None,
            "mad_norm": None,
            "mean_dir": None,
        }

        self.round_state_path = f"runs/{simulation_id}/round_aggregates.jsonl"

    # ---------------- core logic ----------------

    def aggregate_fit(self, server_round, results, failures):
        if not results:
            return None, {}

        w_global = _flatten(self._global_nd)

        deltas, norms, entries = [], [], []
        for client, fit_res in results:
            nds = parameters_to_ndarrays(fit_res.parameters)
            delta = _flatten(nds) - w_global
            deltas.append(delta)
            norms.append(delta.norm())
            entries.append((client, fit_res))

        deltas = torch.stack(deltas)
        norms = torch.stack(norms)
        unit_dirs = deltas / (norms.unsqueeze(1) + 1e-12)

        # order by projection to mean direction
        mean_dir = unit_dirs.mean(dim=0)
        mean_dir = mean_dir / (mean_dir.norm() + 1e-12)
        proj = torch.matmul(unit_dirs, mean_dir)
        order = torch.argsort(proj, descending=True)

        unit_dirs = unit_dirs[order]
        norms = norms[order]
        entries = [entries[i] for i in order.tolist()]

        # -------- intra-round clustering --------
        clusters = _forward_cluster(unit_dirs, theta_s=0.05)

        scaled_local = torch.zeros(len(norms), dtype=torch.bool)
        for idxs in clusters:
            if len(idxs) < 3:
                continue
            scaled_local[idxs] = _robust_z(norms[idxs]) > 3.5

        # -------- inter-round validation --------
        scaled_final = scaled_local.clone()
        if self.round_state["mean_norm"] is not None:
            prev_mean = torch.tensor(self.round_state["mean_norm"])
            prev_mad = torch.tensor(self.round_state["mad_norm"])
            z_prev = (0.6745 * (norms - prev_mean) / (prev_mad + 1e-12)).abs()
            scaled_final &= z_prev > 3.5

        # -------- mitigation --------
        filtered = [
            entry for entry, bad in zip(entries, scaled_final.tolist()) if not bad
        ]

        if len(filtered) < max(2, len(results) // 2):
            filtered = entries
            scaled_final[:] = False

        print(
            f"[Round {server_round}] "
            f"clusters={len(clusters)} "
            f"scaled_local={int(scaled_local.sum())} "
            f"scaled_final={int(scaled_final.sum())}"
        )

        params, metrics = super().aggregate_fit(server_round, filtered, failures)

        # update global reference
        if params is not None:
            self._global_nd = parameters_to_ndarrays(params)

        # update inter-round aggregates
        self.round_state["mean_norm"] = float(norms.mean())
        self.round_state["mad_norm"] = float(_mad(norms))
        self.round_state["mean_dir"] = mean_dir.detach().cpu()

        if self.persist_round_stats:
            os.makedirs(os.path.dirname(self.round_state_path), exist_ok=True)
            with open(self.round_state_path, "a") as f:
                f.write(json.dumps({
                    "round": server_round,
                    "num_clients": len(results),
                    "num_clusters": len(clusters),
                    "scaled_local": int(scaled_local.sum()),
                    "scaled_final": int(scaled_final.sum()),
                    "mean_norm": self.round_state["mean_norm"],
                    "mad_norm": self.round_state["mad_norm"],
                }) + "\n")

        return params, metrics

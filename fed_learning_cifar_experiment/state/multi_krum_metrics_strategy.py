import json
import random
import inspect
import traceback
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import flwr as fl
from flwr.common import FitIns, Parameters, GetPropertiesIns
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.common import parameters_to_ndarrays, ndarrays_to_parameters


# -----------------------------------------------------------------------------
# Import base strategy (version/location tolerant)
# -----------------------------------------------------------------------------
try:
    # If this file lives next to krum_metrics_strategy.py
    from .krum_metrics_strategy import SaveKrumMetricsStrategy
except Exception:
    try:
        # If your project imports it via module path
        from fed_learning_cifar_experiment.krum_metrics_strategy import SaveKrumMetricsStrategy  # type: ignore
    except Exception:
        # Fallback: same directory / PYTHONPATH
        from krum_metrics_strategy import SaveKrumMetricsStrategy  # type: ignore


class SaveMultiKrumMetricsStrategy(SaveKrumMetricsStrategy):
    """
    Multi-Krum variant that mirrors SaveKrumMetricsStrategy except for aggregation:

    - Selection: compute Krum scores for all clients (same as SaveKrumMetricsStrategy)
    - Choose top-k lowest-score clients (k = num_clients_to_select)
    - Aggregate: safe layer-wise average over those k selected parameter sets

    The strategy still provides:
    - identical configure_fit sampling and malicious marking
    - persistent attacker pool behavior
    - krum_selected_cid / krum_ref_delta fields for client-side proxy logic
      (using the *canonical* winner = lowest-score among selected top-k)
    """

    def __init__(
        self,
        *,
        num_clients_to_select: int = 1,
        **kwargs: Any,
    ):
        # Keep base init fully intact
        super().__init__(**kwargs)

        # Multi-Krum only parameter (top-k)
        self.num_clients_to_select = int(num_clients_to_select)

        # Server-only (not sent to clients): track malicious IDs used in the most recent configure_fit
        self._last_round_malicious_ids: List[str] = []

    def configure_fit(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager: ClientManager,
    ) -> List[Tuple[ClientProxy, FitIns]]:
        num_available = len(client_manager.all())
        sample_size, min_num = self.num_fit_clients(num_available)

        all_clients = list(client_manager.all().values())

        if self.attacker_selection_mode == "persistent":
            # ---- Fixed attacker pool ----
            try:
                cfg0 = self.on_fit_config_fn(server_round) if self.on_fit_config_fn else {}
                raw = cfg0.get("malicious-client-ids", "[]")
                malicious_pool = json.loads(raw) if isinstance(raw, str) else list(raw)
            except Exception:
                print(f"[Round {server_round}] Failed to parse malicious client IDs from config. Falling back to empty pool.")
                malicious_pool = []

            malicious_pool = set(str(cid) for cid in malicious_pool)

            attacker_clients = [
                c for c in all_clients if str(c.cid) in malicious_pool
            ][: int(self.num_of_malicious_clients_per_round)]

            attacker_cids = set(c.cid for c in attacker_clients)

            benign_candidates = [
                c for c in all_clients if c.cid not in attacker_cids
            ]

            remaining = max(0, sample_size - len(attacker_clients))
            sampled_benign = random.sample(
                benign_candidates, min(len(benign_candidates), remaining)
            )

            sampled_clients = attacker_clients + sampled_benign
            malicious_ids = [c.cid for c in attacker_clients]

        else:
            # ---- RANDOM attack mode (baseline, unchanged) ----
            sampled_clients = list(client_manager.sample(sample_size, min_num))
            sampled_ids = [c.cid for c in sampled_clients]

            num_malicious = min(
                self.num_of_malicious_clients_per_round, len(sampled_ids)
            )
            malicious_ids = random.sample(sampled_ids, num_malicious)

        sampled_ids = [c.cid for c in sampled_clients]

        # Server-only tracking (no additional leakage to clients)
        self._last_round_malicious_ids = list(map(str, malicious_ids))

        fit_ins_list: List[Tuple[ClientProxy, FitIns]] = []
        print("Sampled clients for round {}: {}".format(server_round, sampled_ids))
        print("Malicious clients for round {}: {}".format(server_round, malicious_ids))
        for client in sampled_clients:
            config = self.on_fit_config_fn(server_round) if self.on_fit_config_fn else {}
            config.update(
                {
                    "current-round": server_round,
                    "sampled_client_ids": json.dumps(sampled_ids),
                    "malicious_client_ids": json.dumps(malicious_ids),
                    "is_malicious": str(client.cid in malicious_ids),
                }
            )

            # Keep same persistent-mode feedback keys, even for Multi-Krum
            if self.attacker_selection_mode == "persistent":
                config["krum_selected_cid"] = self.last_krum_selected_cid
                config["krum_ref_delta"] = (
                    json.dumps(self.last_krum_selected_delta.tolist())
                    if self.last_krum_selected_delta is not None
                    else None
                )

            # Keep prev_global propagation identical
            if self.prev_global_parameters is not None:
                config["prev_global_tensors_hex"] = json.dumps(
                    [t.hex() for t in self.prev_global_parameters.tensors]
                )
                config["prev_global_tensor_type"] = self.prev_global_parameters.tensor_type
            else:
                config["prev_global_tensors_hex"] = "[]"
                config["prev_global_tensor_type"] = "numpy.ndarray"

            fit_ins_list.append((client, FitIns(parameters, config)))

        return fit_ins_list

    @staticmethod
    def _is_float_arr(a: np.ndarray) -> bool:
        return np.issubdtype(a.dtype, np.floating)

    @staticmethod
    def _is_int_or_bool_arr(a: np.ndarray) -> bool:
        return np.issubdtype(a.dtype, np.integer) or np.issubdtype(a.dtype, np.bool_)

    def aggregate_fit(self, rnd: int, results, failures):
        # Keep Flower edge-case behavior consistent with base
        if not results or failures:
            return super().aggregate_fit(rnd, results, failures)

        # ---- Extract client parameter vectors (SAME AS KRUM) ----
        client_ids: List[str] = []
        client_params_nds: List[List[np.ndarray]] = []
        client_updates_flat: List[np.ndarray] = []
        client_proxies: List[ClientProxy] = []

        for client_proxy, fit_res in results:
            nds = parameters_to_ndarrays(fit_res.parameters)
            flat = np.concatenate([np.asarray(p).ravel() for p in nds])
            client_updates_flat.append(flat)
            client_params_nds.append([np.asarray(p) for p in nds])
            client_ids.append(str(client_proxy.cid))
            client_proxies.append(client_proxy)

        X = np.stack(client_updates_flat)
        n = len(X)

        # Byzantine count (version-safe, SAME AS KRUM)
        f = getattr(self, "num_byzantine", None)
        if f is None:
            f = getattr(self, "num_malicious_clients", 0)

        k_neigh = n - int(f) - 2
        if k_neigh <= 0:
            raise RuntimeError(f"Invalid Krum config: n={n}, f={f}")

        # ---- Compute Krum scores (SAME AS KRUM) ----
        scores: Dict[str, float] = {}
        for i in range(n):
            dists = []
            for j in range(n):
                if i == j:
                    continue
                dists.append(float(np.linalg.norm(X[i] - X[j]) ** 2))
            scores[client_ids[i]] = float(sum(sorted(dists)[:k_neigh]))

        # ---- Print scores (SAME FORMAT AS KRUM) ----
        print(f"\n[Round {rnd}][Krum Scores]")
        for cid in sorted(scores, key=scores.get):
            proxy = next(p for p in client_proxies if str(p.cid) == cid)
            pid = self._get_partition_id(proxy)
            print(
                f"  CID={cid:>6} | "
                f"Partition={pid:>3} | "
                f"Score={scores[cid]:.6e}"
            )

        # ---- Multi-Krum selection: top-k lowest-score ----
        k = int(self.num_clients_to_select)
        k = max(1, min(k, n))
        sorted_cids = sorted(scores, key=scores.get)
        selected_cids = sorted_cids[:k]
        selected_idx = [client_ids.index(cid) for cid in selected_cids]

        # Canonical representative for persistent attacker feedback (lowest-score)
        canonical_cid = selected_cids[0]
        canonical_idx = selected_idx[0]

        # Server-side logging only (no new leakage)
        is_attacker_selected = any(str(cid) in set(self._last_round_malicious_ids) for cid in selected_cids)
        print(
            f"[Round {rnd}][Multi-Krum Selected] "
            f"k={k}/{n} | Canonical={canonical_cid} | Selected={selected_cids} | "
            f"Attacker selected={is_attacker_selected}\n"
        )

        # ---- Store Krum-like winner state for next-round proxy logic ----
        self.last_krum_selected_cid = canonical_cid

        # delta = w_canonical - w_prev_global (SAME INTENT AS KRUM)
        if self.prev_global_parameters is not None:
            prev_nds = parameters_to_ndarrays(self.prev_global_parameters)
            prev_flat = np.concatenate([np.asarray(p).ravel() for p in prev_nds])

            curr_flat = client_updates_flat[canonical_idx]
            self.last_krum_selected_delta = curr_flat - prev_flat
        else:
            self.last_krum_selected_delta = None

        # ---- SAFE AGGREGATION: layer-wise mean over selected clients ----
        template = client_params_nds[canonical_idx]
        agg_nds: List[np.ndarray] = []

        for layer_idx, base in enumerate(template):
            base_arr = np.asarray(base)

            if self._is_int_or_bool_arr(base_arr):
                # preserve int/bool layers exactly
                agg_nds.append(base_arr.copy())
                continue

            # float / mixed numeric layers: average in float64, cast back
            acc = np.zeros(base_arr.shape, dtype=np.float64)
            for i in selected_idx:
                layer = np.asarray(client_params_nds[i][layer_idx])
                acc += layer.astype(np.float64, copy=False)

            acc /= float(k)
            agg_nds.append(acc.astype(base_arr.dtype, copy=False))

        new_parameters = ndarrays_to_parameters(agg_nds)

        # ---- Maintain base behavior: prev_global is the model we will send as g_{t-1} next round ----
        self.prev_global_parameters = new_parameters

        return new_parameters, {}

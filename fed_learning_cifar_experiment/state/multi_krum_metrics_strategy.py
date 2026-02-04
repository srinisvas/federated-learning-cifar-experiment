import json
import random
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import flwr as fl
from flwr.common import FitIns, Parameters
from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy

from fed_learning_cifar_experiment.utils.logger import (
    append_distributed_round,
    write_experiment_summary,
)

class SaveMultiKrumMetricsStrategy(fl.server.strategy.FedAvg):

    def __init__(
        self,
        simulation_id: str = "",
        num_clients: int = 0,
        num_rounds: int = 0,
        aggregation_method: str = "",
        backdoor_attack_mode: str = "",
        num_of_malicious_clients: int = 0,
        num_of_malicious_clients_per_round: int = 0,
        num_byzantine: int = 0,
        num_clients_to_select: int = 1,
        normalize_updates: bool = True,
        eps: float = 1e-12,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)

        self.simulation_id = simulation_id
        self.num_clients = num_clients
        self.num_rounds = num_rounds
        self.aggregation_method = aggregation_method
        self.backdoor_attack_mode = backdoor_attack_mode

        self.history = {"round": [], "mta": [], "asr": []}
        self.central_mta_history: List[float] = []
        self.central_asr_history: List[float] = []
        self.final_centralized_mta: Optional[float] = None
        self.final_centralized_asr: Optional[float] = None

        self.num_of_malicious_clients = int(num_of_malicious_clients)
        self.num_of_malicious_clients_per_round = int(num_of_malicious_clients_per_round)

        self.num_byzantine = int(num_byzantine)
        self.num_clients_to_select = int(num_clients_to_select)

        self.normalize_updates = bool(normalize_updates)
        self.eps = float(eps)

        self._global_parameters_for_round: Optional[Parameters] = None

    # -------------------------------------------------------
    # Utility helpers
    # -------------------------------------------------------

    @staticmethod
    def _is_float_arr(a: np.ndarray) -> bool:
        return np.issubdtype(a.dtype, np.floating)

    @staticmethod
    def _is_int_or_bool_arr(a: np.ndarray) -> bool:
        return np.issubdtype(a.dtype, np.integer) or np.issubdtype(a.dtype, np.bool_)

    @classmethod
    def _flatten_float_only(cls, nds: List[np.ndarray]) -> np.ndarray:
        flats: List[np.ndarray] = []
        for a in nds:
            a = np.asarray(a)
            if cls._is_float_arr(a):
                flats.append(a.ravel().astype(np.float64, copy=False))
        if not flats:
            return np.zeros((0,), dtype=np.float64)
        return np.concatenate(flats, axis=0)

    # -------------------------------------------------------
    # configure_fit (unchanged behavior)
    # -------------------------------------------------------

    def configure_fit(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager: ClientManager,
    ):
        self._global_parameters_for_round = parameters

        num_available = len(client_manager.all())
        sample_size, min_num = self.num_fit_clients(num_available)
        sampled_clients = list(client_manager.sample(sample_size, min_num))
        sampled_ids = [c.cid for c in sampled_clients]

        num_malicious = min(self.num_of_malicious_clients_per_round, len(sampled_ids))
        malicious_ids = random.sample(sampled_ids, num_malicious)

        fit_ins_list = []
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
            fit_ins_list.append((client, FitIns(parameters, config)))

        return fit_ins_list

    # -------------------------------------------------------
    # Multi-Krum core (FIXED)
    # -------------------------------------------------------

    def aggregate_fit(self, server_round: int, results, failures):

        if not results:
            return None, {}

        f = self.num_byzantine
        n = len(results)

        if n <= 2 * f + 2:
            print(f"[Round {server_round}] Falling back to FedAvg (n={n}, f={f})")
            aggregated = super().aggregate_fit(server_round, results, failures)
            if aggregated and aggregated[0] is not None:
                self._global_parameters_for_round = aggregated[0]
            return aggregated

        if self._global_parameters_for_round is None:
            aggregated = super().aggregate_fit(server_round, results, failures)
            if aggregated and aggregated[0] is not None:
                self._global_parameters_for_round = aggregated[0]
            return aggregated

        global_nds = [np.asarray(a) for a in parameters_to_ndarrays(self._global_parameters_for_round)]
        global_vec = self._flatten_float_only(global_nds)

        if global_vec.size == 0:
            aggregated = super().aggregate_fit(server_round, results, failures)
            if aggregated and aggregated[0] is not None:
                self._global_parameters_for_round = aggregated[0]
            return aggregated

        client_params_nds = []
        client_update_vecs = []
        client_cids = []

        for client_proxy, fit_res in results:
            nds = [np.asarray(a) for a in parameters_to_ndarrays(fit_res.parameters)]
            vec = self._flatten_float_only(nds)
            update_vec = vec - global_vec

            if self.normalize_updates:
                norm = float(np.linalg.norm(update_vec))
                update_vec = update_vec / (norm + self.eps)

            client_params_nds.append(nds)
            client_update_vecs.append(update_vec)
            client_cids.append(getattr(client_proxy, "cid", "unknown"))

        m = n - f - 2
        dists = np.zeros((n, n), dtype=np.float64)

        for i in range(n):
            ui = client_update_vecs[i]
            for j in range(i + 1, n):
                diff = ui - client_update_vecs[j]
                dist = float(np.dot(diff, diff))
                dists[i, j] = dist
                dists[j, i] = dist

        scores = np.zeros(n, dtype=np.float64)
        for i in range(n):
            row = np.delete(dists[i], i)
            row.sort()
            scores[i] = float(np.sum(row[:m]))

        k = max(1, min(self.num_clients_to_select, n))
        selected_idx = np.argsort(scores)[:k].tolist()
        selected_cids = [client_cids[i] for i in selected_idx]

        print(
            f"[Round {server_round}] Multi-Krum selected k={k}/{n} "
            f"(f={f}, m={m}). Selected CIDs={selected_cids}"
        )

        # -------- SAFE AGGREGATION --------

        template = client_params_nds[selected_idx[0]]
        agg_nds = []

        for layer_idx, base in enumerate(template):
            base = np.asarray(base)

            if self._is_int_or_bool_arr(base):
                agg_nds.append(base.copy())
                continue

            acc = np.zeros(base.shape, dtype=np.float64)
            for i in selected_idx:
                layer = np.asarray(client_params_nds[i][layer_idx])
                acc += layer.astype(np.float64, copy=False)

            acc /= float(k)
            agg_nds.append(acc.astype(base.dtype, copy=False))

        new_parameters = ndarrays_to_parameters(agg_nds)
        self._global_parameters_for_round = new_parameters

        return new_parameters, {}

    # -------------------------------------------------------
    # Evaluation (unchanged)
    # -------------------------------------------------------

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

        if rnd >= self.num_rounds:
            write_experiment_summary(
                simulation_id=self.simulation_id,
                meta={
                    "aggregation": str(self.aggregation_method),
                    "num_rounds": str(self.num_rounds),
                    "num_malicious_clients": str(self.num_clients),
                    "backdoor_attack_mode": str(self.backdoor_attack_mode),
                    "alpha": 0.9,
                },
                final_centralized_mta=self.final_centralized_mta or 0.0,
                final_centralized_asr=self.final_centralized_asr or 0.0,
                dist_mta_history=self.history.get("mta", []),
                dist_asr_history=self.history.get("asr", []),
                central_mta_history=self.central_mta_history,
                central_asr_history=self.central_asr_history,
                notes="",
            )

        return metrics

    def record_centralized_eval(self, rnd, loss, mta, asr):
        self.central_mta_history.append(mta)
        self.central_asr_history.append(asr)
        if rnd == self.num_rounds:
            self.final_centralized_mta = mta
            self.final_centralized_asr = asr

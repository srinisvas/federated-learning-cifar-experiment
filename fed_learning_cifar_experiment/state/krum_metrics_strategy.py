import json
import random
from typing import Dict, List, Optional, Tuple

import flwr as fl
from flwr.common import FitIns, parameters_to_ndarrays, ndarrays_to_parameters

import numpy as np

from fed_learning_cifar_experiment.utils.logger import (
    append_distributed_round,
    write_experiment_summary,
)


class SaveKrumMetricsStrategy(fl.server.strategy.Strategy):
    def __init__(
        self,
        simulation_id: str = "",
        num_clients: int = 0,
        num_rounds: int = 0,
        aggregation_method: str = "Krum",
        backdoor_attack_mode: str = "",
        num_of_malicious_clients: int = 0,
        num_of_malicious_clients_per_round: int = 0,
        num_byzantine: int = 0,

        # ---- FedAvg-compatible args ----
        evaluate_fn=None,
        on_fit_config_fn=None,
        on_evaluate_config_fn=None,
        accept_failures: bool = True,
        initial_parameters=None,

        # ---- sampling knobs (similar to FedAvg) ----
        fraction_fit: float = 1.0,
        min_fit_clients: int = 2,
        min_available_clients: int = 2,
        fraction_evaluate: float = 1.0,
        min_evaluate_clients: int = 2,
    ):
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
        self._cid_to_partition: Dict[str, int] = {}

        # Krum
        self.num_byzantine = int(num_byzantine)

        # FedAvg-compatible callbacks
        self.evaluate_fn = evaluate_fn
        self.on_fit_config_fn = on_fit_config_fn
        self.on_evaluate_config_fn = on_evaluate_config_fn

        self.accept_failures = bool(accept_failures)
        self.initial_parameters = initial_parameters
        self._latest_parameters = initial_parameters

        # Client sampling knobs
        self.fraction_fit = float(fraction_fit)
        self.min_fit_clients = int(min_fit_clients)
        self.min_available_clients = int(min_available_clients)

        self.fraction_evaluate = float(fraction_evaluate)
        self.min_evaluate_clients = int(min_evaluate_clients)

    def initialize_parameters(self, client_manager):
        return self.initial_parameters

    def evaluate(self, server_round: int, parameters):
        if self.evaluate_fn is None:
            return None
        return self.evaluate_fn(server_round, parameters)

    def configure_fit(self, server_round: int, parameters, client_manager):
        self._latest_parameters = parameters

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

    def aggregate_fit(self, server_round, results, failures):
        if not results:
            return None, {}
        if failures and not self.accept_failures:
            return None, {}

        w_global = parameters_to_ndarrays(self._latest_parameters)

        updates = []
        client_params = []

        for _, fit_res in results:
            w_client = parameters_to_ndarrays(fit_res.parameters)
            upd = [wc - wg for wc, wg in zip(w_client, w_global)]
            updates.append(upd)
            client_params.append(w_client)

        chosen_idx = self._krum_select_index(updates, self.num_byzantine)
        chosen_weights = client_params[chosen_idx]

        new_params = ndarrays_to_parameters(chosen_weights)
        self._latest_parameters = new_params

        return new_params, {"krum_selected_client_index": chosen_idx}

    def aggregate_evaluate(self, rnd, results, failures):
        if not results:
            return None, {}

        # same metrics logging behavior as your FedAvg
        mta_vals = [res.metrics.get("mta", 0.0) for _, res in results]
        asr_vals = [res.metrics.get("asr", 0.0) for _, res in results]

        avg_mta = sum(mta_vals) / len(mta_vals) if mta_vals else 0.0
        avg_asr = sum(asr_vals) / len(asr_vals) if asr_vals else 0.0

        self.history["round"].append(rnd)
        self.history["mta"].append(avg_mta)
        self.history["asr"].append(avg_asr)

        print(f"[Round {rnd}] MTA={avg_mta:.4f}, ASR={avg_asr:.4f}")

        # dist loss if present
        loss_vals = [(res.num_examples, res.loss) for _, res in results]
        total = sum(n for n, _ in loss_vals)
        dist_loss = sum(n * loss for n, loss in loss_vals) / total if total > 0 else None

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
                    "krum_num_byzantine": str(self.num_byzantine),
                },
                final_centralized_mta=self.final_centralized_mta or 0.0,
                final_centralized_asr=self.final_centralized_asr or 0.0,
                dist_mta_history=self.history.get("mta", []),
                dist_asr_history=self.history.get("asr", []),
                central_mta_history=self.central_mta_history,
                central_asr_history=self.central_asr_history,
                notes="",
            )

        return dist_loss, {"mta": avg_mta, "asr": avg_asr}

    def record_centralized_eval(self, rnd, loss, mta, asr):
        self.central_mta_history.append(mta)
        self.central_asr_history.append(asr)
        if rnd == self.num_rounds:
            self.final_centralized_mta = mta
            self.final_centralized_asr = asr

    # -------- helpers --------

    def num_fit_clients(self, num_available_clients: int) -> Tuple[int, int]:
        sample_size = int(num_available_clients * self.fraction_fit)
        sample_size = max(sample_size, self.min_fit_clients)
        return sample_size, self.min_available_clients

    @staticmethod
    def _flatten_update(update: List[np.ndarray]) -> np.ndarray:
        return np.concatenate([u.reshape(-1).astype(np.float32, copy=False) for u in update])

    def _krum_select_index(self, updates: List[List[np.ndarray]], f: int) -> int:
        n = len(updates)
        if n == 1:
            return 0

        f = int(f)
        if n < (2 * f + 3):
            # fallback: closest to mean
            flat = [self._flatten_update(u) for u in updates]
            mean = np.mean(np.stack(flat, axis=0), axis=0)
            d = [np.sum((v - mean) ** 2) for v in flat]
            return int(np.argmin(d))

        flat = [self._flatten_update(u) for u in updates]

        dist = np.zeros((n, n), dtype=np.float64)
        for i in range(n):
            for j in range(i + 1, n):
                dij = np.sum((flat[i] - flat[j]) ** 2)
                dist[i, j] = dij
                dist[j, i] = dij

        m = n - f - 2
        scores = []
        for i in range(n):
            dists = np.sort(dist[i])[1:]
            scores.append(np.sum(dists[:m]))

        return int(np.argmin(scores))
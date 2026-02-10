import json
import random
import inspect
from typing import Dict, List, Optional, Tuple, Any

import flwr as fl
from flwr.common import FitIns, Parameters
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy

from fed_learning_cifar_experiment.utils.logger import (
    append_distributed_round,
    write_experiment_summary,
)


class SaveKrumMetricsStrategy(fl.server.strategy.Krum):
    """
    Krum clone of SaveFedAvgMetricsStrategy:
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
        **kwargs: Any,
    ):
        # Store your extra metadata (not part of Flower Strategy API)
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

        self.num_of_malicious_clients = num_of_malicious_clients
        self.num_of_malicious_clients_per_round = num_of_malicious_clients_per_round
        self._cid_to_partition: Dict[str, int] = {}

        # --------- Compatibility layer for Flower Krum constructor ---------
        # Some versions use num_byzantine, some use num_malicious_clients.
        # Your server_app passes num_byzantine=2, so map if needed.
        if "num_byzantine" in kwargs and "num_malicious_clients" not in kwargs:
            # If Krum expects num_malicious_clients but caller gave num_byzantine
            # we will remap later based on signature.
            pass

        # Filter kwargs to only those accepted by your installed Flower Krum.__init__
        sig = inspect.signature(fl.server.strategy.Krum.__init__)
        accepted = set(sig.parameters.keys())

        forwarded: Dict[str, Any] = {}
        for k, v in kwargs.items():
            if k in accepted:
                forwarded[k] = v

        # If user passed num_byzantine but constructor expects num_malicious_clients (or vice versa)
        if "num_byzantine" in kwargs and "num_byzantine" not in accepted:
            if "num_malicious_clients" in accepted and "num_malicious_clients" not in forwarded:
                forwarded["num_malicious_clients"] = kwargs["num_byzantine"]

        if "num_malicious_clients" in kwargs and "num_malicious_clients" not in accepted:
            if "num_byzantine" in accepted and "num_byzantine" not in forwarded:
                forwarded["num_byzantine"] = kwargs["num_malicious_clients"]

        # In some older versions, Krum might not accept evaluate_fn/initial_parameters
        # The filtering above will automatically drop them if unsupported.
        super().__init__(**forwarded)

        # If evaluate_fn wasn't forwarded (unsupported by base Krum), keep it anyway.
        # Flower may still call Strategy.evaluate (implemented in base classes).
        self._evaluate_fn_fallback = kwargs.get("evaluate_fn", None)
        self._initial_parameters_fallback = kwargs.get("initial_parameters", None)
        # Track previous global model (g_{t-1}) to send to clients
        self.prev_global_parameters: Optional[Parameters] = None

    def configure_fit(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager: ClientManager,
    ) -> List[Tuple[ClientProxy, FitIns]]:
        num_available = len(client_manager.all())
        sample_size, min_num = self.num_fit_clients(num_available)
        sampled_clients = list(client_manager.sample(sample_size, min_num))
        sampled_ids = [c.cid for c in sampled_clients]

        # Randomly pick malicious clients from the sampled list
        num_malicious = min(self.num_of_malicious_clients_per_round, len(sampled_ids))
        malicious_ids = random.sample(sampled_ids, num_malicious)

        fit_ins_list: List[Tuple[ClientProxy, FitIns]] = []
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
            dist_mta = self.history.get("mta", [])
            dist_asr = self.history.get("asr", [])

            write_experiment_summary(
                simulation_id=self.simulation_id,
                meta={
                    "aggregation": str(self.aggregation_method),
                    "num_rounds": str(self.num_rounds),
                    # Keeping your original behavior (even if it's a naming mismatch)
                    "num_malicious_clients": str(self.num_clients),
                    "backdoor_attack_mode": str(self.backdoor_attack_mode),
                    "alpha": 0.9,
                },
                final_centralized_mta=self.final_centralized_mta or 0.0,
                final_centralized_asr=self.final_centralized_asr or 0.0,
                dist_mta_history=dist_mta,
                dist_asr_history=dist_asr,
                central_mta_history=self.central_mta_history,
                central_asr_history=self.central_asr_history,
                notes="",
            )

        return metrics

    def aggregate_fit(self, rnd, results, failures):
        aggregated = super().aggregate_fit(rnd, results, failures)

        if aggregated and aggregated[0] is not None:
            selected_params = aggregated[0]

            # Identify which client was selected by Krum
            selected_cid = None
            for client_proxy, fit_res in results:
                if fit_res.parameters.tensors == selected_params.tensors:
                    selected_cid = client_proxy.cid
                    break

            is_attacker_selected = (
                    selected_cid is not None
                    and selected_cid in getattr(self, "_last_round_malicious_ids", set())
            )

            print(
                f"[Round {rnd}][Krum] "
                f"Selected CID={selected_cid}, "
                f"Attacker selected={is_attacker_selected}"
            )

            # Keep your existing behavior
            self.prev_global_parameters = selected_params

        return aggregated

    def record_centralized_eval(self, rnd: int, loss: float, mta: float, asr: float) -> None:
        self.central_mta_history.append(mta)
        self.central_asr_history.append(asr)
        if rnd == self.num_rounds:
            self.final_centralized_mta = mta
            self.final_centralized_asr = asr
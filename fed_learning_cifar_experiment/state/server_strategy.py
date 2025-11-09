import json
import random

import flwr as fl
from flwr.common import FitIns
from fed_learning_cifar_experiment.utils.logger import (
    append_distributed_round,
    write_experiment_summary,
)


class SaveFedAvgMetricsStrategy(fl.server.strategy.FedAvg):
    def __init__(self,
                 simulation_id: str = "",
                 num_clients: int = 0,
                 num_rounds: int = 0,
                 aggregation_method: str = "",
                 backdoor_attack_mode: str = "",
                 **kwargs):
        super().__init__(**kwargs)
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

    def configure_fit(self, server_round: int, parameters, client_manager):
        num_available_clients = len(client_manager.all())
        sample_size, min_num_clients = self.num_fit_clients(num_available_clients)

        sampled_clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )
        sampled_clients = list(sampled_clients)

        sampled_ids = [c.cid for c in sampled_clients]
        print(f"[Round {server_round}] Sampled client IDs: {sampled_ids}")

        num_malicious = 1
        malicious_ids = random.sample(sampled_ids, num_malicious)
        print(f"[Round {server_round}] Selected malicious clients: {malicious_ids}")

        config = self.on_fit_config_fn(server_round) if self.on_fit_config_fn else {}
        config.update({
            "current-round": server_round,
            "sampled_client_ids": json.dumps(sampled_ids),
            "malicious_client_ids": json.dumps(malicious_ids),
        })
        fit_ins = FitIns(parameters, config)

        return [(client, fit_ins) for client in sampled_clients]

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

        # If last round: also write final summary
        if rnd >= self.num_rounds:
            dist_mta = self.history.get("mta", [])
            dist_asr = self.history.get("asr", [])

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
                dist_mta_history=dist_mta,
                dist_asr_history=dist_asr,
                central_mta_history=self.central_mta_history,
                central_asr_history=self.central_asr_history,
                notes=""
            )

        return metrics

    def record_centralized_eval(self, rnd, loss, mta, asr):
        self.central_mta_history.append(mta)
        self.central_asr_history.append(asr)
        if rnd == self.num_rounds:
            self.final_centralized_mta = mta
            self.final_centralized_asr = asr

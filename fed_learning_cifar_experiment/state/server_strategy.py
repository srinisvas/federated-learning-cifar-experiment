import json
import random

import flwr as fl
from flwr.common import FitIns, GetPropertiesIns
from fed_learning_cifar_experiment.utils.logger import (
    append_distributed_round,
    write_experiment_summary,
)

import numpy as np
import torch
from torch.nn.utils import parameters_to_vector
from flwr.common import parameters_to_ndarrays
from fed_learning_cifar_experiment.task import (
    get_resnet_cnn_model, set_weights, load_data, train
)


class SaveFedAvgMetricsStrategy(fl.server.strategy.FedAvg):
    def __init__(self,
                 simulation_id: str = "",
                 num_clients: int = 0,
                 num_rounds: int = 0,
                 aggregation_method: str = "",
                 backdoor_attack_mode: str = "",
                 num_of_malicious_clients = 0,
                 num_of_malicious_clients_per_round = 0,
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
        self.num_of_malicious_clients = num_of_malicious_clients
        self.num_of_malicious_clients_per_round = num_of_malicious_clients_per_round
        self._cid_to_partition = {}
        self.prev_global_parameters = None
        self._last_round_malicious_ids: list = []

    def configure_fit(self, server_round: int, parameters, client_manager):
        num_available = len(client_manager.all())
        sample_size, min_num = self.num_fit_clients(num_available)
        sampled_clients = list(client_manager.sample(sample_size, min_num))
        sampled_ids = [c.cid for c in sampled_clients]

        num_malicious = min(self.num_of_malicious_clients_per_round, len(sampled_ids))
        malicious_ids = random.sample(sampled_ids, num_malicious)
        self._last_round_malicious_ids = list(map(str, malicious_ids))

        # --- Build shared_ref_deltas (same logic as Krum strategy) ---
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        nds = parameters_to_ndarrays(parameters)
        model_tmp = get_resnet_cnn_model()
        set_weights(model_tmp, nds)
        model_tmp.to(device)
        init_vec = parameters_to_vector(model_tmp.parameters()).detach().cpu()

        ref_partition_ids = random.sample(range(self.num_clients), 6)
        ref_deltas = []
        for pid in ref_partition_ids:
            train_loader, _ = load_data(
                partition_id=pid,
                num_partitions=self.num_clients,
                alpha_val=0.9,
                backdoor_enabled=False,
            )
            net_ref = get_resnet_cnn_model()
            set_weights(net_ref, nds)
            net_ref.to(device)
            lr = random.choice([0.003, 0.004, 0.005])
            epochs = random.choice([1, 2])
            _, vec = train(net_ref, train_loader, epochs, device, lr)
            delta = (vec - init_vec).cpu().numpy()
            ref_deltas.append(delta)

        ref_deltas = np.stack(ref_deltas)
        norms = np.linalg.norm(ref_deltas, axis=1)
        median_norm = float(np.median(norms))
        # --- End ref_deltas block ---

        print(f"[Round {server_round}] Sampled: {sampled_ids} | Malicious: {malicious_ids}")

        fit_ins_list = []
        for client in sampled_clients:
            config = self.on_fit_config_fn(server_round) if self.on_fit_config_fn else {}
            config.update({
                "current-round": server_round,
                "sampled_client_ids": json.dumps(sampled_ids),
                "malicious_client_ids": json.dumps(malicious_ids),
                "is_malicious": str(client.cid in malicious_ids),
                "shared_ref_deltas": json.dumps(ref_deltas.tolist()),
                "shared_ref_median_norm": median_norm,
            })
            # Send prev global so clients can compute delta from it (same as Krum)
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

    def aggregate_fit(self, rnd, results, failures):
        aggregated = super().aggregate_fit(rnd, results, failures)
        if aggregated is not None:
            agg_params, _ = aggregated
            self.prev_global_parameters = agg_params
        return aggregated
import json
import random
import inspect
import traceback
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from flwr.common import parameters_to_ndarrays

import flwr as fl
from flwr.common import FitIns, Parameters
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.common import GetPropertiesIns

from fed_learning_cifar_experiment.utils.logger import (
    append_distributed_round,
    write_experiment_summary,
)

from flwr.common import parameters_to_ndarrays
from fed_learning_cifar_experiment.task import (
    get_resnet_cnn_model, set_weights, load_data, train
)
import torch
from torch.nn.utils import parameters_to_vector


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
        self.attacker_selection_mode = kwargs.get("attacker_selection_mode", "random").lower()
        # Track previous global model (g_{t-1}) to send to clients
        self.prev_global_parameters: Optional[Parameters] = None
        self.last_krum_selected_cid: Optional[str] = None
        self.last_krum_selected_delta: Optional[np.ndarray] = None

    def _get_partition_id(self, client: ClientProxy) -> int:
        if client.cid in self._cid_to_partition:
            return self._cid_to_partition[client.cid]

        try:
            res = client.get_properties(GetPropertiesIns(config={}), timeout=5.0, group_id=None)

            if res is None:
                raise RuntimeError("get_properties returned None")

            if not hasattr(res, "properties"):
                raise RuntimeError(f"Invalid GetPropertiesRes: {res}")

            if "partition_id" not in res.properties:
                raise KeyError(f"'partition_id' missing in properties: {res.properties}")

            pid_raw = res.properties["partition_id"]
            pid = int(pid_raw)

        except Exception as e:
            print("=" * 80)
            print("[ERROR] Failed to fetch partition_id")
            print(f"CID            : {client.cid}")
            print(f"Client type    : {type(client)}")
            print(f"Exception type : {type(e).__name__}")
            print(f"Exception msg  : {e}")
            print("Stack trace:")
            traceback.print_exc()
            print("=" * 80)
            pid = -1

        self._cid_to_partition[client.cid] = pid
        return pid

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

            _, vec = train(
                net_ref,
                train_loader,
                epochs,
                device,
                lr,
            )

            delta = (vec - init_vec).cpu().numpy()
            ref_deltas.append(delta)

        ref_deltas = np.stack(ref_deltas)

        centroid = np.mean(ref_deltas, axis=0)
        norms = np.linalg.norm(ref_deltas, axis=1)
        median_norm = float(np.median(norms))

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
                    "shared_ref_deltas": json.dumps(ref_deltas.tolist()),
                    "shared_ref_median_norm": median_norm,

                }
            )

            if self.attacker_selection_mode == "persistent":
                config["krum_selected_cid"] = self.last_krum_selected_cid
                config["krum_ref_delta"] = (
                    json.dumps(self.last_krum_selected_delta.tolist())
                    if self.last_krum_selected_delta is not None
                    else None
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
        # Let Flower handle edge cases first
        if not results or failures:
            return super().aggregate_fit(rnd, results, failures)

        # ---- Extract updates ----
        client_ids = []
        client_updates = []
        client_proxies = []

        for client_proxy, fit_res in results:
            nds = parameters_to_ndarrays(fit_res.parameters)
            flat = np.concatenate([p.flatten() for p in nds])
            client_updates.append(flat)
            client_ids.append(client_proxy.cid)
            client_proxies.append(client_proxy)

        X = np.stack(client_updates)
        n = len(X)

        # ---- Compute reference vectors ----
        centroid = np.mean(X, axis=0)
        centroid_dir = centroid / (np.linalg.norm(centroid) + 1e-12)

        norms = np.linalg.norm(X, axis=1)

        def cosine(a, b):
            return np.dot(a, b) / ((np.linalg.norm(a) + 1e-12) * (np.linalg.norm(b) + 1e-12))

        # Byzantine count (version-safe)
        f = getattr(self, "num_byzantine", None)
        if f is None:
            f = getattr(self, "num_malicious_clients", 0)

        k = n - f - 2
        if k <= 0:
            raise RuntimeError(f"Invalid Krum config: n={n}, f={f}")

        # ---- Compute Krum scores ----
        scores = {}
        for i in range(n):
            dists = []
            for j in range(n):
                if i == j:
                    continue
                dists.append(np.linalg.norm(X[i] - X[j]) ** 2)
            score = sum(sorted(dists)[:k])
            scores[client_ids[i]] = score

        # ---- Print scores ----

        print(f"\n[Round {rnd}][Krum Scores + Geometry]")
        for cid in sorted(scores, key=scores.get):
            i = client_ids.index(cid)
            proxy = next(p for p in client_proxies if p.cid == cid)
            pid = self._get_partition_id(proxy)

            vec = X[i]

            norm = norms[i]
            cos_centroid = cosine(vec, centroid)
            dist_centroid = np.linalg.norm(vec - centroid)

            # Projection on centroid direction
            proj_mag = np.dot(vec, centroid_dir)
            proj_vec = proj_mag * centroid_dir

            # Orthogonal deviation
            orth_dev = np.linalg.norm(vec - proj_vec)

            dists = np.linalg.norm(X - vec, axis=1)
            dists = np.delete(dists, i)

            nearest = np.sort(dists)[:k]

            nearest_mean = np.mean(nearest)
            nearest_max = np.max(nearest)
            nearest_min = np.min(nearest)

            # distance to previous global delta
            if self.last_krum_selected_delta is not None:
                prev = self.last_krum_selected_delta
                cos_prev = cosine(vec, prev)
                dist_prev = np.linalg.norm(vec - prev)
            else:
                cos_prev = 0.0
                dist_prev = 0.0

            print(
                f"CID={cid:>6} | "
                f"Partition={pid:>3} | "
                f"Score={scores[cid]:.6e} | "
                f"Norm={norm:.4f} | "
                f"CosCentroid={cos_centroid:.4f} | "
                f"DistCentroid={dist_centroid:.4f} | "
                f"ProjCentroid={proj_mag:.4f} | "
                f"OrthDev={orth_dev:.4f} | "
                f"NNmean={nearest_mean:.4f} | "
                f"NNmax={nearest_max:.4f} | "
                f"NNmin={nearest_min:.4f} | "
                f"CosPrev={cos_prev:.4f} | "
                f"DistPrev={dist_prev:.4f}"
            )

        # ---- Select winner ----
        selected_cid = min(scores, key=scores.get)
        selected_idx = client_ids.index(selected_cid)
        selected_params = results[selected_idx][1].parameters

        winner_vec = X[selected_idx]

        print(f"\n[Round {rnd}] Distance to Krum Winner")

        for i, cid in enumerate(client_ids):
            proxy = next(p for p in client_proxies if p.cid == cid)
            pid = self._get_partition_id(proxy)

            dist = np.linalg.norm(X[i] - winner_vec)
            cos = cosine(X[i], winner_vec)

            print(
                f"CID={cid:>6} | "
                f"Partition={pid:>3} | "
                f"DistWinner={dist:.4f} | "
                f"CosWinner={cos:.4f}"
            )

        # ---- Store Krum winner state (for next round) ----
        self.last_krum_selected_cid = selected_cid

        # Compute delta = w_selected - w_prev_global
        if self.prev_global_parameters is not None:
            prev_nds = parameters_to_ndarrays(self.prev_global_parameters)
            curr_nds = parameters_to_ndarrays(selected_params)

            prev_flat = np.concatenate([p.flatten() for p in prev_nds])
            curr_flat = np.concatenate([p.flatten() for p in curr_nds])

            self.last_krum_selected_delta = curr_flat - prev_flat
        else:
            self.last_krum_selected_delta = None

        selected_partition = self._get_partition_id(client_proxies[selected_idx])

        print(
            f"[Round {rnd}][Krum Selected] "
            f"CID={selected_cid}, Partition={selected_partition}\n"
        )

        # ---- Maintain your existing behavior ----
        self.prev_global_parameters = selected_params

        # ---- Return in Flower-compatible format ----
        # Use Flower’s expected return type: (Parameters, metrics)
        return selected_params, {}

    def record_centralized_eval(self, rnd: int, loss: float, mta: float, asr: float) -> None:
        self.central_mta_history.append(mta)
        self.central_asr_history.append(asr)
        if rnd == self.num_rounds:
            self.final_centralized_mta = mta
            self.final_centralized_asr = asr
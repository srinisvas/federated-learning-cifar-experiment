import json
import random
import traceback
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import flwr as fl
from flwr.common import FitIns, GetPropertiesIns, Parameters, parameters_to_ndarrays
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy

import torch
from torch.nn.utils import parameters_to_vector

from fed_learning_cifar_experiment.utils.logger import (
    append_distributed_round,
    write_experiment_summary,
    append_per_update_features,
)
from fed_learning_cifar_experiment.utils.metrics_extractor import (
    ParamRegistry,
    nds_to_trainable_state_dict,
    state_dict_delta,
    flat_param_vec_to_per_key_dict,
    mean_delta_per_key,
    extract_per_update_features,
    build_leave_one_out_means,
    build_stage_projection_bases,
    ref_deltas_flat_to_per_key_list,
)
from fed_learning_cifar_experiment.task import (
    get_resnet_cnn_model, set_weights, load_data, train,
)


class SaveFedAvgMetricsStrategy(fl.server.strategy.FedAvg):
    """
    FedAvg with the full per-update structural feature extraction pipeline,
    matching SaveKrumMetricsStrategy exactly.

    Differences from the Krum variant:
      - Aggregation: weighted average of all client updates (standard FedAvg).
      - 'krum_score'  -> L2 distance from the client delta to the round-mean
        delta (a meaningful, Krum-free analog; keeps the CSV schema identical).
      - 'krum_rank'   -> rank by that distance (1 = closest to mean).
      - 'selected_by_aggregator' -> 1 for every client (all contribute to FedAvg).
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
        super().__init__(**kwargs)

        self.simulation_id = simulation_id
        self.num_clients = num_clients
        self.num_rounds = num_rounds
        self.aggregation_method = aggregation_method
        self.backdoor_attack_mode = backdoor_attack_mode
        self.num_of_malicious_clients = num_of_malicious_clients
        self.num_of_malicious_clients_per_round = num_of_malicious_clients_per_round

        self.history: Dict[str, list] = {"round": [], "mta": [], "asr": []}
        self.central_mta_history: List[float] = []
        self.central_asr_history: List[float] = []
        self.final_centralized_mta: Optional[float] = None
        self.final_centralized_asr: Optional[float] = None

        self._cid_to_partition: Dict[str, int] = {}

        # Previous global parameters (g_{t-1}) — forwarded to clients for
        # Neurotoxin benign-grad approximation and constrain-and-scale.
        self.prev_global_parameters: Optional[Parameters] = None

        # Per-round state: captured in configure_fit, consumed in aggregate_fit.
        self._param_registry: Optional[ParamRegistry] = None
        self._round_global_parameters: Optional[Parameters] = None
        self._round_ref_deltas_flat: Optional[np.ndarray] = None
        self._round_attack_mode: str = "none"
        self._round_attack_type: str = "none"
        self._round_sampled_ids: List[str] = []
        self._round_malicious_ids_set: set = set()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_partition_id(self, client: ClientProxy) -> int:
        if client.cid in self._cid_to_partition:
            return self._cid_to_partition[client.cid]

        try:
            res = client.get_properties(
                GetPropertiesIns(config={}), timeout=5.0, group_id=None
            )
            if res is None:
                raise RuntimeError("get_properties returned None")
            if not hasattr(res, "properties"):
                raise RuntimeError(f"Invalid GetPropertiesRes: {res}")
            if "partition_id" not in res.properties:
                raise KeyError(f"'partition_id' missing in properties: {res.properties}")
            pid = int(res.properties["partition_id"])
        except Exception as e:
            print("=" * 80)
            print("[ERROR] Failed to fetch partition_id")
            print(f"CID            : {client.cid}")
            print(f"Exception type : {type(e).__name__}")
            print(f"Exception msg  : {e}")
            traceback.print_exc()
            print("=" * 80)
            pid = -1

        self._cid_to_partition[client.cid] = pid
        return pid

    # ------------------------------------------------------------------
    # configure_fit — mirrors Krum: builds ref deltas, sends prev global
    # ------------------------------------------------------------------

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

        num_malicious = min(self.num_of_malicious_clients_per_round, len(sampled_ids))
        malicious_ids = random.sample(sampled_ids, num_malicious)

        self._round_malicious_ids_set = set(str(x) for x in malicious_ids)
        self._round_sampled_ids = [str(x) for x in sampled_ids]

        print(f"Sampled clients for round {server_round}: {sampled_ids}")
        print(f"Malicious clients for round {server_round}: {malicious_ids}")

        # ---- Build reference clean deltas (same as Krum strategy) ----
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        nds = parameters_to_ndarrays(parameters)

        model_tmp = get_resnet_cnn_model()
        set_weights(model_tmp, nds)
        model_tmp.to(device)
        init_vec = parameters_to_vector(model_tmp.parameters()).detach().cpu()

        ref_partition_ids = random.sample(range(self.num_clients), 6)
        ref_deltas: List[np.ndarray] = []

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
            ref_deltas.append((vec - init_vec).cpu().numpy())

        ref_deltas_np = np.stack(ref_deltas)           # [6, D_params]
        median_norm = float(np.median(np.linalg.norm(ref_deltas_np, axis=1)))

        # ---- Build per-client FitIns ----
        fit_ins_list: List[Tuple[ClientProxy, FitIns]] = []
        for client in sampled_clients:
            config = self.on_fit_config_fn(server_round) if self.on_fit_config_fn else {}
            config.update({
                "current-round": server_round,
                "sampled_client_ids": json.dumps(sampled_ids),
                "malicious_client_ids": json.dumps(malicious_ids),
                "is_malicious": str(client.cid in malicious_ids),
                "shared_ref_deltas": json.dumps(ref_deltas_np.tolist()),
                "shared_ref_median_norm": median_norm,
            })

            # Forward previous global so clients can compute benign-grad approx.
            if self.prev_global_parameters is not None:
                config["prev_global_tensors_hex"] = json.dumps(
                    [t.hex() for t in self.prev_global_parameters.tensors]
                )
                config["prev_global_tensor_type"] = self.prev_global_parameters.tensor_type
            else:
                config["prev_global_tensors_hex"] = "[]"
                config["prev_global_tensor_type"] = "numpy.ndarray"

            fit_ins_list.append((client, FitIns(parameters, config)))

        # ---- Stash per-round state for aggregate_fit ----
        if self._param_registry is None:
            tmp_model = get_resnet_cnn_model()
            self._param_registry = ParamRegistry(tmp_model)
            print(
                f"[Extractor] ParamRegistry built: "
                f"{len(self._param_registry.entries)} trainable params, "
                f"{self._param_registry.total_trainable_params} elements, "
                f"{len(self._param_registry.state_dict_keys)} state_dict keys"
            )

        # Global model going INTO this round — used by feature extractor as the
        # baseline against which client deltas are computed.
        self._round_global_parameters = parameters
        self._round_ref_deltas_flat = ref_deltas_np

        cfg_template = self.on_fit_config_fn(server_round) if self.on_fit_config_fn else {}
        self._round_attack_mode = str(cfg_template.get("backdoor-attack-mode", "none"))
        self._round_attack_type = str(cfg_template.get("backdoor-attack-type", "none"))

        return fit_ins_list

    # ------------------------------------------------------------------
    # aggregate_fit — run FedAvg aggregation, then extract features
    # ------------------------------------------------------------------

    def aggregate_fit(
        self,
        rnd: int,
        results: List[Tuple[ClientProxy, Any]],
        failures,
    ):
        if not results:
            return super().aggregate_fit(rnd, results, failures)

        # FedAvg weighted average — let the parent handle it.
        aggregated_params, aggregated_metrics = super().aggregate_fit(
            rnd, results, failures
        )

        # Store the new global for the next configure_fit call.
        if aggregated_params is not None:
            self.prev_global_parameters = aggregated_params

        # ---- Compute FedAvg-analog scores (dist to round mean delta) ----
        client_ids = [cp.cid for cp, _ in results]
        client_proxies = [cp for cp, _ in results]

        flat_updates: List[np.ndarray] = []
        for _, fit_res in results:
            nds = parameters_to_ndarrays(fit_res.parameters)
            flat_updates.append(np.concatenate([p.flatten() for p in nds]))
        X = np.stack(flat_updates)                              # [n, D_model]

        round_mean_flat = np.mean(X, axis=0)
        dist_to_mean = np.linalg.norm(X - round_mean_flat, axis=1)

        rank_order = np.argsort(dist_to_mean)
        ranks = np.empty_like(rank_order)
        ranks[rank_order] = np.arange(1, len(rank_order) + 1)

        scores = {client_ids[i]: float(dist_to_mean[i]) for i in range(len(client_ids))}
        rank_of = {client_ids[i]: int(ranks[i]) for i in range(len(client_ids))}

        print(f"\n[Round {rnd}][FedAvg Dist-to-Mean]")
        for i, cid in enumerate(client_ids):
            pid = self._get_partition_id(client_proxies[i])
            is_mal = str(cid) in self._round_malicious_ids_set
            print(
                f"CID={cid:>6} | Partition={pid:>3} | "
                f"DistToMean={dist_to_mean[i]:.4f} | "
                f"Rank={ranks[i]} | Malicious={is_mal}"
            )

        # ---- Extract per-update structural features ----
        try:
            self._extract_and_log_features(
                rnd=rnd,
                results=results,
                client_ids=client_ids,
                client_proxies=client_proxies,
                scores=scores,
                rank_of=rank_of,
            )
        except Exception as e:
            print(
                f"[Extractor][Round {rnd}] Feature extraction failed: "
                f"{type(e).__name__}: {e}"
            )
            traceback.print_exc()

        return aggregated_params, aggregated_metrics

    # ------------------------------------------------------------------
    # aggregate_evaluate
    # ------------------------------------------------------------------

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
            self.simulation_id, rnd, avg_mta, avg_asr, dist_loss, self.num_clients,
        )

        if rnd >= self.num_rounds:
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
                dist_mta_history=self.history.get("mta", []),
                dist_asr_history=self.history.get("asr", []),
                central_mta_history=self.central_mta_history,
                central_asr_history=self.central_asr_history,
                notes="",
            )

        return metrics

    def record_centralized_eval(self, rnd: int, loss: float, mta: float, asr: float) -> None:
        self.central_mta_history.append(mta)
        self.central_asr_history.append(asr)
        if rnd == self.num_rounds:
            self.final_centralized_mta = mta
            self.final_centralized_asr = asr

    # ------------------------------------------------------------------
    # _extract_and_log_features
    # ------------------------------------------------------------------

    def _extract_and_log_features(
        self,
        rnd: int,
        results: List[Tuple[ClientProxy, Any]],
        client_ids: List[str],
        client_proxies: List[ClientProxy],
        scores: Dict[str, float],
        rank_of: Dict[str, int],
    ) -> None:
        """
        Compute per-update structural features for every client this round and
        append one CSV row per client via append_per_update_features.

        Schema is identical to the Krum variant:
          krum_score          -> L2 distance from client delta to round-mean delta
          krum_rank           -> rank by that distance (1 = most central)
          selected_by_aggregator -> always 1 (all clients contribute to FedAvg)
        """
        if self._param_registry is None:
            print(f"[Extractor][Round {rnd}] No registry, skipping.")
            return

        registry = self._param_registry

        global_params = self._round_global_parameters
        if global_params is None:
            global_params = self.prev_global_parameters
        if global_params is None:
            print(f"[Extractor][Round {rnd}] No global parameters, skipping.")
            return

        global_nds = parameters_to_ndarrays(global_params)
        global_sd_train = nds_to_trainable_state_dict(global_nds, registry.state_dict_keys)

        # ---- Per-client delta dicts (trainable params only) ----
        per_client: List[Dict[str, Any]] = []
        for client_proxy, fit_res in results:
            cid = client_proxy.cid
            try:
                client_nds = parameters_to_ndarrays(fit_res.parameters)
                client_sd = nds_to_trainable_state_dict(client_nds, registry.state_dict_keys)
                delta_sd = state_dict_delta(client_sd, global_sd_train)
            except Exception as e:
                print(f"[Extractor][Round {rnd}] Skipping CID={cid}: {e}")
                continue
            per_client.append({
                "cid": cid,
                "proxy": client_proxy,
                "fit_res": fit_res,
                "delta_sd": delta_sd,
            })

        if not per_client:
            return

        # ---- Round-mean delta and leave-one-out means ----
        round_mean_sd = mean_delta_per_key(c["delta_sd"] for c in per_client)
        loo_means = build_leave_one_out_means([c["delta_sd"] for c in per_client])
        for i, c in enumerate(per_client):
            c["loo_mean"] = loo_means[i]

        # ---- Reference clean delta ----
        reference_delta_per_key: Optional[Dict] = None
        ref_deltas_per_key_list: Optional[List] = None
        if self._round_ref_deltas_flat is not None and len(self._round_ref_deltas_flat) > 0:
            try:
                ref_mean_flat = torch.from_numpy(
                    self._round_ref_deltas_flat.mean(axis=0)
                ).float()
                if ref_mean_flat.numel() == registry.total_trainable_params:
                    reference_delta_per_key = flat_param_vec_to_per_key_dict(
                        ref_mean_flat, registry
                    )
                    ref_deltas_per_key_list = ref_deltas_flat_to_per_key_list(
                        self._round_ref_deltas_flat, registry
                    )
                else:
                    print(
                        f"[Extractor][Round {rnd}] ref_deltas size mismatch: "
                        f"{ref_mean_flat.numel()} != {registry.total_trainable_params}"
                    )
            except Exception as e:
                print(f"[Extractor][Round {rnd}] reference build failed: {e}")

        # ---- Per-stage projection bases ----
        stage_proj_bases: Optional[Dict] = None
        stage_proj_ref_means: Optional[Dict] = None
        if ref_deltas_per_key_list is not None:
            try:
                stage_proj_bases, stage_proj_ref_means = build_stage_projection_bases(
                    ref_deltas_per_key_list, registry, k=3
                )
            except Exception as e:
                print(f"[Extractor][Round {rnd}] projection basis fit failed: {e}")

        # ---- Per-client feature extraction and CSV write ----
        for entry in per_client:
            cid = entry["cid"]
            proxy = entry["proxy"]
            fit_res = entry["fit_res"]
            delta_sd = entry["delta_sd"]
            loo_mean = entry["loo_mean"]

            partition_id = self._get_partition_id(proxy)
            is_malicious = str(cid) in self._round_malicious_ids_set
            client_metrics = dict(getattr(fit_res, "metrics", {}) or {})

            effective_attack_type = self._round_attack_type if is_malicious else "none"

            metadata = {
                "simulation_id": self.simulation_id,
                "round": rnd,
                "cid": str(cid),
                "partition_id": int(partition_id) if partition_id is not None else -1,
                "malicious_flag": int(is_malicious),
                "attack_mode": self._round_attack_mode,
                "attack_type": effective_attack_type,
                "local_data_size": int(getattr(fit_res, "num_examples", 0)),
                "local_epochs": client_metrics.get("local_epochs"),
                "local_lr": client_metrics.get("local_lr"),
                "scale_factor": client_metrics.get("scale_factor"),
                # All clients contribute equally in FedAvg.
                "selected_by_aggregator": 1,
                # Repurposed Krum fields — see class docstring.
                "krum_score": float(scores.get(cid, float("nan"))),
                "krum_rank": int(rank_of.get(cid, -1)),
                "dirichlet_alpha": 0.9,
                "target_label": 2,
                "aggregation_method": str(self.aggregation_method),
                "num_clients": int(self.num_clients),
            }

            try:
                features = extract_per_update_features(
                    client_delta_per_key=delta_sd,
                    round_mean_delta_per_key=round_mean_sd,
                    registry=registry,
                    reference_delta_per_key=reference_delta_per_key,
                    leave_one_out_mean_per_key=loo_mean,
                    stage_projection_bases=stage_proj_bases,
                    stage_projection_ref_means=stage_proj_ref_means,
                )
            except Exception as e:
                print(f"[Extractor][Round {rnd}][CID={cid}] feature compute failed: {e}")
                continue

            try:
                append_per_update_features(metadata=metadata, features=features)
            except Exception as e:
                print(f"[Extractor][Round {rnd}][CID={cid}] CSV write failed: {e}")
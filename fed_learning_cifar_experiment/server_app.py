import json
import os
import random

import torch
from flwr.common import Context, ndarrays_to_parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg

from fed_learning_cifar_experiment.state.fedavg_cluster_defense import SaveFedAvgMetricsClusterDefenseStrategy
from fed_learning_cifar_experiment.state.krum_metrics_strategy import SaveKrumMetricsStrategy
from fed_learning_cifar_experiment.state.multi_krum_metrics_strategy import SaveMultiKrumMetricsStrategy
from fed_learning_cifar_experiment.state.server_strategy import SaveFedAvgMetricsStrategy

from fed_learning_cifar_experiment.utils.evaluate_attack import get_evaluate_fn
from fed_learning_cifar_experiment.task import get_weights, get_resnet_cnn_model, load_test_data_for_eval


def server_fn(context: Context):
    # Read from config
    num_rounds = context.run_config["num-server-rounds"]
    fraction_fit = context.run_config["fraction-fit"]
    num_clients = context.run_config["num-clients"]
    simulation_id = context.run_config.get("simulation-id")
    aggregation_method = context.run_config.get("aggregation-method", "fedavg").lower()
    backdoor_attack_mode = context.run_config.get("backdoor-attack-mode", "none").lower()
    backdoor_attack_type = context.run_config.get("backdoor-attack-type", "train-and-scale").lower()
    num_of_malicious_clients = context.run_config.get("num-malicious-clients", 0)
    num_of_malicious_clients_per_round = context.run_config.get("num-malicious-clients-per-round", 1)
    attacker_selection_mode = context.run_config.get("attacker-selection-mode", "random").lower()
    malicious_client_id = context.run_config.get("malicious-client-id", 2)

    # Optional fixed backdoor partition list.  Pass as a JSON array in run-config:
    #   backdoor-partition-ids='[5,23,67,12,44,78,3,91,50,16]'
    # If omitted, the strategy draws 10 at random (still fixed for the run).
    _raw_bdp = context.run_config.get("backdoor-partition-ids", None)
    backdoor_partition_ids = json.loads(_raw_bdp) if _raw_bdp else None
    # backdoor_rounds must be defined before on_fit_config_fn closes over it,
    # regardless of attack type — otherwise any unhandled type causes a NameError.
    backdoor_rounds = json.dumps([])  # safe default; overridden below

    if backdoor_attack_mode == "global-random-attack":
        if backdoor_attack_type == "train-and-scale":
            backdoor_rounds = json.dumps([1])
        elif backdoor_attack_type == "constrain-and-scale":
            backdoor_rounds = json.dumps(
                [1, 6, 11, 16, 21, 26, 31, 36, 41, 46, 51, 56, 61, 66, 71, 76, 81, 86, 91, 96]
            )
        elif backdoor_attack_type == "neurotoxin":
            # Inject every 5 rounds: enough events to measure inter-attack ASR decay.
            backdoor_rounds = json.dumps(list(range(1, int(num_rounds) + 1, 5)))
        else:
            backdoor_rounds = json.dumps([1])
            print(f"[WARNING] Unhandled backdoor_attack_type '{backdoor_attack_type}' "
                  f"for global-random-attack; defaulting backdoor_rounds=[1].")

    if backdoor_attack_mode == "global-attack-first" and backdoor_attack_type == "neurotoxin":
        backdoor_rounds = json.dumps([1, 11, 21, 31, 41, 51, 61, 71, 81, 91])
    # Initialize model parameter

    model = get_resnet_cnn_model()
    print("Attack Selection Mode:", attacker_selection_mode)
    if torch.cuda.is_available() and os.path.exists("pretrained_cifar_bw8.pth"):
        print("Loading pretrained global model...")
        model.load_state_dict(torch.load("pretrained_cifar_bw8.pth", map_location="cpu"))
    #model_nd_arrays = get_weights(get_resnet_cnn_model())
    model_nd_arrays = get_weights(model)
    #model_nd_arrays = get_weights(get_basic_cnn_model())
    parameters = ndarrays_to_parameters(model_nd_arrays)

    testing_data = load_test_data_for_eval(batch_size=64)

    def on_fit_config_fn(server_round: int):
        on_fit_config = {}
        if backdoor_attack_mode == "none":
            on_fit_config = {
                "backdoor-attack-mode": "none",
            }
        elif backdoor_attack_mode == "global-random-attack":
            on_fit_config = {
                "backdoor-attack-mode": "global-random-attack",
                "current-round": server_round,
                "backdoor-rounds": backdoor_rounds,
                "backdoor-attack-type": backdoor_attack_type,
            }
        elif backdoor_attack_mode ==  "global-attack-first":
            on_fit_config = {
                "backdoor-attack-mode": "global-attack-first",
                "backdoor-attack-type": backdoor_attack_type,
            }
        elif backdoor_attack_mode == "per-round-attack":
            on_fit_config = {
                "backdoor-attack-mode": "per-round-attack",
                "backdoor-attack-type": backdoor_attack_type
            }
            if context.run_config.get("attack-selection-mode", "random").lower() == "persistent":
                on_fit_config["malicious-client-ids"] = context.run_config.get(
                    "malicious-client-ids", "[]"
                )
        return on_fit_config

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if aggregation_method == "fedavg":
        strategy = SaveFedAvgMetricsStrategy(
            fraction_fit=0.1,
            fraction_evaluate=0.1,
            min_fit_clients=10,
            min_available_clients=100,
            evaluate_fn=get_evaluate_fn(model=get_resnet_cnn_model().to(device), test_data=testing_data),
            initial_parameters=parameters,
            on_fit_config_fn=on_fit_config_fn,
            simulation_id=simulation_id,
            num_clients=num_clients,
            num_rounds=num_rounds,
            aggregation_method=aggregation_method,
            backdoor_attack_mode=backdoor_attack_mode,
            num_of_malicious_clients=num_of_malicious_clients,
            num_of_malicious_clients_per_round=num_of_malicious_clients_per_round,
            backdoor_partition_ids=backdoor_partition_ids,
        )
    elif aggregation_method == "fedavg-cluster-defense":
        strategy = SaveFedAvgMetricsClusterDefenseStrategy(
            fraction_fit=0.1,
            fraction_evaluate=0.1,
            min_fit_clients=10,
            min_available_clients=100,
            evaluate_fn=get_evaluate_fn(model=get_resnet_cnn_model().to(device), test_data=testing_data),
            initial_parameters=parameters,
            on_fit_config_fn=on_fit_config_fn,
            simulation_id = simulation_id,
            num_clients = num_clients,
            num_rounds=num_rounds,
            aggregation_method=aggregation_method,
            backdoor_attack_mode = backdoor_attack_mode,
            num_of_malicious_clients = num_of_malicious_clients,
            num_of_malicious_clients_per_round = num_of_malicious_clients_per_round
        )

    elif aggregation_method == "krum":
        strategy = SaveKrumMetricsStrategy(
            fraction_fit=0.1,
            fraction_evaluate=0.1,
            min_fit_clients=10,
            min_available_clients=100,
            evaluate_fn=get_evaluate_fn(model=get_resnet_cnn_model().to(device), test_data=testing_data),
            initial_parameters=parameters,
            on_fit_config_fn=on_fit_config_fn,
            simulation_id=simulation_id,
            num_clients=num_clients,
            num_rounds=num_rounds,
            aggregation_method=aggregation_method,
            backdoor_attack_mode=backdoor_attack_mode,
            num_of_malicious_clients=num_of_malicious_clients,
            num_of_malicious_clients_per_round=num_of_malicious_clients_per_round,
            attacker_selection_mode = attacker_selection_mode,
            num_byzantine=int(num_of_malicious_clients_per_round),  # or num_of_malicious_clients_per_round
        )

    elif aggregation_method == "multikrum":
        strategy = SaveMultiKrumMetricsStrategy(
            fraction_fit=0.1,
            fraction_evaluate=0.1,
            min_fit_clients=10,
            min_available_clients=100,
            evaluate_fn=get_evaluate_fn(model=get_resnet_cnn_model().to(device), test_data=testing_data),
            initial_parameters=parameters,
            on_fit_config_fn=on_fit_config_fn,
            simulation_id=simulation_id,
            num_clients=num_clients,
            num_rounds=num_rounds,
            aggregation_method=aggregation_method,
            backdoor_attack_mode=backdoor_attack_mode,
            num_of_malicious_clients=num_of_malicious_clients,
            num_of_malicious_clients_per_round=num_of_malicious_clients_per_round,

            num_byzantine=int(num_of_malicious_clients_per_round),  # f
            num_clients_to_select=5,  # k
            normalize_updates=True,  # keep this on
        )

    config = ServerConfig(num_rounds=num_rounds)

    return ServerAppComponents(strategy=strategy, config=config)

app = ServerApp(server_fn=server_fn)
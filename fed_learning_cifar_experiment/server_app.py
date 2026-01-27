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
    malicious_client_id = context.run_config.get("malicious-client-id", 2)
    if backdoor_attack_mode == "global-random-attack" and backdoor_attack_type == "train-and-scale":
        hardcoded_rounds = [1, 2]
        #backdoor_rounds = json.dumps(random.sample(range(1, num_rounds + 1), num_of_malicious_clients))
        backdoor_rounds = json.dumps(hardcoded_rounds)
    if backdoor_attack_mode == "global-random-attack" and backdoor_attack_type == "constrain-and-scale":
        hardcoded_rounds = [1, 11, 21, 31, 41]
        #backdoor_rounds = json.dumps(random.sample(range(1, num_rounds + 1), num_of_malicious_clients))
        backdoor_rounds = json.dumps(hardcoded_rounds)
    # Initialize model parameter

    model = get_resnet_cnn_model()
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
            simulation_id = simulation_id,
            num_clients = num_clients,
            num_rounds = num_rounds,
            aggregation_method = aggregation_method,
            backdoor_attack_mode = backdoor_attack_mode,
            num_of_malicious_clients = num_of_malicious_clients,
            num_of_malicious_clients_per_round = num_of_malicious_clients_per_round
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

            num_byzantine=2,  # or num_of_malicious_clients_per_round
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

            num_byzantine=2,  # f
            num_clients_to_select=5,  # k
            normalize_updates=True,  # keep this on
        )

    config = ServerConfig(num_rounds=num_rounds)

    return ServerAppComponents(strategy=strategy, config=config)

app = ServerApp(server_fn=server_fn)

import json
from collections import OrderedDict

import torch

from flwr.client import ClientApp, NumPyClient
from flwr.common import Context, ConfigRecord
from torch.nn.utils import parameters_to_vector, vector_to_parameters

from fed_learning_cifar_experiment.task import (get_weights, load_data, set_weights, test, train, get_resnet_cnn_model,
                                                get_basic_cnn_model, train_backdoor)
from fed_learning_cifar_experiment.utils.evaluate_attack import evaluate_asr


# Define Flower Client and client_fn

class FlowerClient(NumPyClient):
    def __init__(self, net, local_epochs, context: Context):
        self.test_set = None
        self.training_set = None
        self.net = net
        self.context = context
        self.local_epochs = local_epochs
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.net.to(self.device)
        self.client_state = context.state
        self.partition_id = str(self.context.node_config.get("partition-id"))

        if "num_backdoor_counts" not in self.client_state.config_records:
            self.client_state.config_records["num_backdoor_counts"] = ConfigRecord({"count": 0})

    def get_properties(self, config):
        return {"partition_id": self.partition_id}

    def fit(self, parameters, config):
        set_weights(self.net, parameters)
        init_state = {k: v.cpu().clone() for k, v in self.net.state_dict().items()}
        init_vec = parameters_to_vector(self.net.parameters()).detach().cpu().clone()
        net_copy = get_resnet_cnn_model()
        set_weights(net_copy, parameters)
        net_copy.to(self.device)

        attack_mode = config.get("backdoor-attack-mode", "none").lower()
        partition_id = self.context.node_config["partition-id"]
        num_partitions = self.context.node_config["num-partitions"]
        num_clients_total = int(self.context.run_config.get("num-clients", 100))
        fraction_fit = float(self.context.run_config.get("fraction-fit", 0.1))
        sampled_clients = 10
        malicious_partitions = json.loads(config.get("malicious_partitions", "[]"))
        my_partition = str(self.context.node_config["partition-id"])
        sampled_client_ids = json.loads(config.get("sampled_client_ids", "[]"))
        malicious_client_ids = json.loads(config.get("malicious_client_ids", "[]"))
        current_round = config.get("current-round", "N/A")
        print(f"[Client {partition_id}] Current round: {current_round}")
        print(f"[Client {partition_id}] Sampled clients this round: {sampled_client_ids}")
        print(f"[Client {partition_id}] Malicious clients this round: {malicious_client_ids}")
        learning_rate = 0.1
        is_attacking_round = False

        if attack_mode == "global-attack-first" and my_partition in malicious_partitions:
            num_malicious_clients = int(config.get("num-malicious-clients", 1))
            attack_count = self.client_state.config_records["num_backdoor_counts"]["count"]
            if attack_count < num_malicious_clients:
                is_attacking_round = True
                print("Global Attack In Progress #Client ID: " + str(partition_id))
                self.training_set, _ = load_data(partition_id, num_partitions, alpha_val=0.9, backdoor_enabled=True)
                self.client_state.config_records["num_backdoor_counts"]["count"] += 1
                self.local_epochs = 100
                learning_rate = 0.01
                print("Incremented attack count to " + str(self.client_state.config_records["num_backdoor_counts"]))
            else:
                self.training_set, _ = load_data(partition_id, num_partitions, alpha_val=0.9)
        elif attack_mode == "global-random-attack" and my_partition in malicious_partitions:
            backdoor_rounds = json.loads(config["backdoor-rounds"])
            print("Rounds Selected for Backdoor:" + str(backdoor_rounds))
            current_round = config["current-round"]
            if current_round in backdoor_rounds:
                print("Global Random Attack Injected #Client ID: " + str(partition_id) + " #Round: " + str(current_round))
                is_attacking_round = True
                self.training_set, _ = load_data(partition_id, num_partitions, alpha_val=0.9, backdoor_enabled=True)
                self.local_epochs = 100
                learning_rate = 0.01
            else:
                self.training_set, _ = load_data(partition_id, num_partitions, alpha_val=0.9)
        elif attack_mode == "per-round-attack":
            backdoor_client_ids = json.loads(config["backdoor-client-ids"])
            if my_partition in malicious_partitions:
                print("Backdoor Attack Injected #Client ID: " + str(partition_id))
                is_attacking_round = True
                self.training_set, _ = load_data(partition_id, num_partitions, alpha_val=0.9, backdoor_enabled=True)
                self.local_epochs = 100
                learning_rate = 0.01
            else:
                self.training_set, _ = load_data(partition_id, num_partitions, alpha_val=0.9)
        else:
            self.training_set, _ = load_data(partition_id, num_partitions, alpha_val=0.9)


        if is_attacking_round:
            # --- DEBUG METRICS START ---
            print(f"\n--- DEBUG: ATTACKER (Client {partition_id}, Round {config.get('current-round', 'N/A')}) ---")
            print(f"--- DEBUG: Initial global model norm (||G_t||): {init_vec.norm().item():.6f}")
            train_loss, final_vec = train_backdoor(
                self.net,
                self.training_set,
                self.local_epochs,
                self.device,
                learning_rate
            )
            print(f"--- DEBUG: Attacker model norm (||X||): {final_vec.norm().item():.6f}")
            delta = final_vec.cpu() - init_vec.cpu()
            print(f"--- DEBUG: Update delta norm (||X - G_t||): {delta.norm().item():.6f}")

            eta = 10 #(num_clients_total / (sampled_clients * fraction_fit))
            print(f"--- DEBUG: Scaling factor (eta): {eta}")
            scaled_vec = init_vec + eta * delta
            scaled_delta_norm = (eta * delta).norm().item()
            print(f"--- DEBUG: Scaled delta norm (||eta * (X - G_t)||): {scaled_delta_norm:.6f}")
            print(f"--- DEBUG: Final model norm (||G_t + eta*delta||): {scaled_vec.norm().item():.6f}")
            print(f"--- DEBUG: ATTACK FINISHED ---\n")
            vector_to_parameters(scaled_vec.to(self.device), self.net.parameters())
            return get_weights(self.net), len(self.training_set.dataset), {"train_loss": train_loss}
        else:
            train_loss, final_vec = train(
                self.net,
                self.training_set,
                self.local_epochs,
                self.device,
                learning_rate
            )
            return get_weights(self.net), len(self.training_set.dataset), {"train_loss": train_loss}

    def evaluate(self, parameters, config):
        partition_id = self.context.node_config["partition-id"]
        num_partitions = self.context.node_config["num-partitions"]
        _, self.test_set = load_data(partition_id, num_partitions, alpha_val=0.9)
        set_weights(self.net, parameters)
        loss, accuracy = test(self.net, self.test_set, self.device)
        # Evaluate Client Side Attack Success Rate
        client_side_asr = evaluate_asr(self.net, self.test_set, target_label=2, device=self.device)
        return loss, len(self.test_set.dataset), {"mta": accuracy, "asr": client_side_asr}

def client_fn(context: Context):

    # Load the default model
    #net = get_basic_cnn_model()
    #Load the updated model(ResNet)
    net = get_resnet_cnn_model()
    local_epochs = context.run_config["local-epochs"]
    partition_id = context.node_config["partition-id"]
    client = FlowerClient(net, local_epochs, context)

    client.cid = str(partition_id)
    print(f"Initialized client with partition ID: {partition_id} (CID set to {client.cid})")
    # Return Client instance
    return client.to_client()
    #return FlowerClient(net, local_epochs, context).to_client()

# Flower ClientApp
app = ClientApp(
    client_fn,
)

import json
from collections import OrderedDict

import torch

from flwr.client import ClientApp, NumPyClient
from flwr.common import Context, ConfigRecord
from torch.nn.utils import parameters_to_vector, vector_to_parameters

from fed_learning_cifar_experiment.task import (get_weights, load_data, set_weights, test, train, get_resnet_cnn_model,
                                                get_basic_cnn_model)
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

        if "num_backdoor_counts" not in self.client_state.config_records:
            self.client_state.config_records["num_backdoor_counts"] = ConfigRecord({"count": 0})

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
        num_clients_total = int(self.context.run_config.get("num-clients", 10))
        fraction_fit = float(self.context.run_config.get("fraction-fit", 1.0))
        sampled_clients = max(1, int(round(fraction_fit * num_clients_total)))
        learning_rate = 0.1
        is_attacking_round = False

        if attack_mode == "global-attack-first" and partition_id == config["malicious-client-id"]:
            num_malicious_clients = int(config.get("num-malicious-clients", 1))
            attack_count = self.client_state.config_records["num_backdoor_counts"]["count"]
            if attack_count < num_malicious_clients:
                is_attacking_round = True
                print("Global Attack In Progress #Client ID: " + str(partition_id))
                self.training_set, _ = load_data(partition_id, num_partitions, alpha_val=0.9, backdoor_enabled=True)
                self.client_state.config_records["num_backdoor_counts"]["count"] += 1
                self.local_epochs = 10
                learning_rate = 0.02
                print("Incremented attack count to " + str(self.client_state.config_records["num_backdoor_counts"]))
            else:
                self.training_set, _ = load_data(partition_id, num_partitions, alpha_val=0.9)
        elif attack_mode == "global-random-attack" and partition_id == config["malicious-client-id"]:
            backdoor_rounds = json.loads(config["backdoor-rounds"])
            print("Rounds Selected for Backdoor:" + str(backdoor_rounds))
            current_round = config["current-round"]
            if current_round in backdoor_rounds:
                print("Global Random Attack Injected #Client ID: " + str(partition_id) + " #Round: " + str(current_round))
                is_attacking_round = True
                self.training_set, _ = load_data(partition_id, num_partitions, alpha_val=0.9, backdoor_enabled=True)
                self.local_epochs = 10
                learning_rate = 0.02
            else:
                self.training_set, _ = load_data(partition_id, num_partitions, alpha_val=0.9)
        elif attack_mode == "per-round-attack":
            backdoor_client_ids = json.loads(config["backdoor-client-ids"])
            if partition_id in backdoor_client_ids:
                print("Backdoor Attack Injected #Client ID: " + str(partition_id))
                is_attacking_round = True
                self.training_set, _ = load_data(partition_id, num_partitions, alpha_val=0.9, backdoor_enabled=True)
                self.local_epochs = 10
                learning_rate = 0.02
            else:
                self.training_set, _ = load_data(partition_id, num_partitions, alpha_val=0.9)
        else:
            self.training_set, _ = load_data(partition_id, num_partitions, alpha_val=0.9)

        train_loss, final_vec = train(
            self.net,
            self.training_set,
            self.local_epochs,
            self.device,
            learning_rate
        )
        """
        # --- diagnostics start (paste here immediately after train(...) returns) ---
        # train_loss, final_vec = train(...)
        # sanity: shapes & finiteness
        print("DIAG: train_loss", train_loss)
        print("DIAG: init_vec shape", init_vec.shape, "final_vec shape", final_vec.shape)
        print("DIAG: init_vec finite:", torch.isfinite(init_vec).all().item(),
              "final_vec finite:", torch.isfinite(final_vec).all().item())

        # basic vector stats
        init_norm = init_vec.norm().item()
        final_norm = final_vec.norm().item()
        delta = final_vec - init_vec
        delta_norm = delta.norm().item()
        print(f"DIAG: ||init||={init_norm:.6e}  ||final||={final_norm:.6e}  ||delta||={delta_norm:.6e}")

        # sample values (first 10)
        print("DIAG: init_vec[:10]", init_vec[:10].cpu().numpy())
        print("DIAG: final_vec[:10]", final_vec[:10].cpu().numpy())
        print("DIAG: delta[:10]", delta[:10].cpu().numpy())

        # per-layer quick checks using state_dicts
        sd_init = {k: v.clone() for k, v in self.net.state_dict().items()}
        sd_after = net_copy.state_dict()  # net_copy is attacker-trained net
        keys_of_interest = ["conv1.weight", "fc.weight", "bn1.running_mean", "bn1.running_var"]
        for k in keys_of_interest:
            if k in sd_init and k in sd_after:
                a = sd_init[k]
                b = sd_after[k]
                print(f"DIAG: key={k} init mean/std {a.mean().item():.6e}/{a.std().item():.6e} | "
                      f"malicious mean/std {b.mean().item():.6e}/{b.std().item():.6e} | "
                      f"diff_norm {(b.view(-1) - a.view(-1)).norm().item():.6e}")
            else:
                print("DIAG: key missing:", k)

        # configuration diagnostics: clients, malicious, local epochs, lr
        m = int(config.get("num-malicious-clients", 1))
        num_clients_total = int(self.context.run_config.get("num-clients", 10))
        sampled_clients = int(self.context.run_config.get("num-clients", num_clients_total))  # fallback
        eta = float(num_clients_total) / float(max(1, m))
        print("DIAG: num_clients_total", num_clients_total, "num_malicious", m,
              "sampled_clients", sampled_clients, "eta(n/m)", eta,
              "local_epochs", self.local_epochs, "learning_rate", learning_rate)

        # check for NaNs/Infs per-layer in net_copy (attacker net)
        malformed = False
        for k, v in sd_after.items():
            if not torch.isfinite(v).all():
                print("DIAG: non-finite in attacker state:", k)
                malformed = True
                break
        print("DIAG: attacker state finite:", (not malformed))

        # quick scaled-norm preview (do not apply to model, just compute)
        scaled_vec_preview = init_vec + eta * delta
        print("DIAG: ||scaled_vec_preview||:", scaled_vec_preview.norm().item(), "scaled_vec_preview[:10]:",
              scaled_vec_preview[:10].cpu().numpy())
        # --- diagnostics end ---
        """
        if is_attacking_round:
            delta = final_vec.cpu() - init_vec.cpu()
            m = int(config.get("num-malicious-clients", 1))
            eta = sampled_clients / max(1, m)
            scaled_vec = init_vec + eta * delta
            vector_to_parameters(scaled_vec.to(self.device), self.net.parameters())

            return get_weights(self.net), len(self.training_set.dataset), {"train_loss": train_loss}
        else:
            return get_weights(self.net), len(self.training_set.dataset), {"train_loss": train_loss}

    def evaluate(self, parameters, config):
        partition_id = self.context.node_config["partition-id"]
        num_partitions = self.context.node_config["num-partitions"]
        _, self.test_set = load_data(partition_id, num_partitions, alpha_val=0.9)
        set_weights(self.net, parameters)
        loss, accuracy = test(self.net, self.test_set, self.device)
        # Evaluate Client Side Attack Success Rate
        _, backdoored_test_set = load_data(partition_id, num_partitions, alpha_val=0.9, backdoor_enabled=True)
        client_side_asr = evaluate_asr(self.net, backdoored_test_set, target_label=2, device=self.device)
        return loss, len(self.test_set.dataset), {"mta": accuracy, "asr": client_side_asr}

def client_fn(context: Context):

    # Load the default model
    #net = get_basic_cnn_model()
    #Load the updated model(ResNet)
    net = get_resnet_cnn_model()

    local_epochs = context.run_config["local-epochs"]

    # Return Client instance
    return FlowerClient(net, local_epochs, context).to_client()

# Flower ClientApp
app = ClientApp(
    client_fn,
)

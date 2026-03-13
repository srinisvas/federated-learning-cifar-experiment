import json
from collections import OrderedDict

import torch

from flwr.client import ClientApp, NumPyClient
from flwr.common import Context, ConfigRecord
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from flwr.common import Parameters, parameters_to_ndarrays

from fed_learning_cifar_experiment.task import (
    get_weights, load_data, set_weights, test, train, get_resnet_cnn_model,
    get_basic_cnn_model, train_backdoor, krum_safe_scale,
    train_constrain_and_scale_krum_proxy, build_reference_clean_deltas
)
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
        self.prev_global_vec = None

        if "num_backdoor_counts" not in self.client_state.config_records:
            self.client_state.config_records["num_backdoor_counts"] = ConfigRecord({"count": 0})
        if "malicious_centroid" not in self.client_state.config_records:
            self.client_state.config_records["malicious_centroid"] = ConfigRecord({
                "vec": [],
                "alpha": 0.9,
            })

    def get_properties(self, config):
        return {"partition_id": str(self.context.node_config["partition-id"])}


    def fit(self, parameters, config):
        krum_ref_delta = None
        if "krum_ref_delta" in config and config["krum_ref_delta"] is not None:
            krum_ref_delta = torch.tensor(
                json.loads(config["krum_ref_delta"]),
                dtype=torch.float32,
            )
        benign_epochs = self.local_epochs
        attack_epochs = self.local_epochs
        set_weights(self.net, parameters)
        init_state = {k: v.cpu().clone() for k, v in self.net.state_dict().items()}
        init_vec = parameters_to_vector(self.net.parameters()).detach().cpu().clone()

        net_ref = get_resnet_cnn_model()
        set_weights(net_ref, parameters)

        prev_global_vec = None
        tensors_hex = json.loads(config.get("prev_global_tensors_hex", "[]"))
        if tensors_hex:
            prev_params = Parameters(
                tensors=[bytes.fromhex(h) for h in tensors_hex],
                tensor_type=config.get("prev_global_tensor_type", "numpy.ndarray"),
            )
            prev_nds = parameters_to_ndarrays(prev_params)
            tmp = get_resnet_cnn_model()
            set_weights(tmp, prev_nds)
            prev_global_vec = parameters_to_vector(tmp.parameters()).detach().cpu().clone()
        else:
            prev_global_vec = None

        # Ensure we always have a prev_global_vec after round 1
        # If None, treat previous as current for stable behavior
        net_copy = get_resnet_cnn_model()
        set_weights(net_copy, parameters)
        net_copy.to(self.device)
        attack_mode = config.get("backdoor-attack-mode", "none").lower()
        attack_type = config.get("backdoor-attack-type", "train-and-scale").lower()
        partition_id = self.context.node_config["partition-id"]
        num_partitions = self.context.node_config["num-partitions"]
        num_clients_total = int(self.context.run_config.get("num-clients", 100))
        fraction_fit = float(self.context.run_config.get("fraction-fit", 0.1))
        sampled_clients = 10

        sampled_client_ids = json.loads(config.get("sampled_client_ids", "[]"))
        malicious_client_ids = json.loads(config.get("malicious_client_ids", "[]"))
        is_malicious = str(config.get("is_malicious", "False")).lower() == "true"

        current_round = config.get("current-round", "N/A")
        partition_id = self.context.node_config["partition-id"]

        #print(f"[Client {partition_id}] Round {current_round}")
        #print(f"[Client {partition_id}] Is malicious? {is_malicious}")
        #print(f"[Client {partition_id}] Sampled clients: {sampled_client_ids}")
        #print(f"[Client {partition_id}] Malicious clients: {malicious_client_ids}")

        learning_rate = 0.005
        is_attacking_round = False

        if attack_mode == "global-attack-first":
            num_malicious_clients = int(config.get("num-malicious-clients", 3))
            attack_count = self.client_state.config_records["num_backdoor_counts"]["count"]
            if attack_count < num_malicious_clients:
                is_attacking_round = True
                print("Global Attack In Progress #Client ID: " + str(partition_id))
                self.training_set, _ = load_data(partition_id, num_partitions, alpha_val=0.9, backdoor_enabled=True)
                self.client_state.config_records["num_backdoor_counts"]["count"] += 1
                attack_epochs = 40
                learning_rate = 0.01
                #print("Incremented attack count to " + str(self.client_state.config_records["num_backdoor_counts"]))
            else:
                self.training_set, _ = load_data(partition_id, num_partitions, alpha_val=0.9)
        elif attack_mode == "global-random-attack" and is_malicious:
            backdoor_rounds = json.loads(config["backdoor-rounds"])
            print("Rounds Selected for Backdoor:" + str(backdoor_rounds))
            current_round = config["current-round"]
            if current_round in backdoor_rounds:
                print("Global Random Attack Injected #Client ID: " + str(partition_id) + " #Round: " + str(current_round))
                is_attacking_round = True
                self.training_set, _ = load_data(partition_id, num_partitions, alpha_val=0.9, backdoor_enabled=True)
                attack_epochs = 40
                learning_rate = 0.01
            else:
                self.training_set, _ = load_data(partition_id, num_partitions, alpha_val=0.9)

        elif attack_mode == "per-round-attack":

            if is_malicious:
                print(f"[Round {current_round}] Per-Round Attack Injected #Client ID: {partition_id}")
                is_attacking_round = True
                self.training_set, _ = load_data(
                    partition_id,
                    num_partitions,
                    alpha_val=0.9,
                    backdoor_enabled=True
                )
                attack_epochs = 5
                learning_rate = 0.005
            else:
                self.training_set, _ = load_data(
                    partition_id,
                    num_partitions,
                    alpha_val=0.9
                )

        else:
            self.training_set, _ = load_data(partition_id, num_partitions, alpha_val=0.9)


        if is_attacking_round:

            if attack_type == "train-and-scale":
                # --- DEBUG METRICS START ---
                # print(f"\n--- DEBUG: ATTACKER (Client {partition_id}, Round {config.get('current-round', 'N/A')}) ---")
                # print(f"--- DEBUG: Initial global model norm (||G_t||): {init_vec.norm().item():.6f}")
                attack_epochs = 40
                train_loss, final_vec = train_backdoor(
                    self.net,
                    self.training_set,
                    attack_epochs,
                    self.device,
                    learning_rate
                )
                # print(f"--- DEBUG: Attacker model norm (||X||): {final_vec.norm().item():.6f}")
                delta = final_vec.cpu() - init_vec.cpu()
                # print(f"--- DEBUG: Update delta norm (||X - G_t||): {delta.norm().item():.6f}")

                eta = 10  # (num_clients_total / (sampled_clients * fraction_fit))
                # print(f"--- DEBUG: Scaling factor (eta): {eta}")
                scaled_vec = init_vec + eta * delta
                scaled_delta_norm = (eta * delta).norm().item()
                # print(f"--- DEBUG: Scaled delta norm (||eta * (X - G_t)||): {scaled_delta_norm:.6f}")
                # print(f"--- DEBUG: Final model norm (||G_t + eta*delta||): {scaled_vec.norm().item():.6f}")
                # print(f"--- DEBUG: ATTACK FINISHED ---\n")
                vector_to_parameters(scaled_vec.to(self.device), self.net.parameters())
                self.prev_global_vec = init_vec.clone()
                return get_weights(self.net), len(self.training_set.dataset), {"train_loss": train_loss}

            else:
                clean_training_set, _ = load_data(
                    partition_id,
                    num_partitions,
                    alpha_val=0.9,
                    backdoor_enabled=False
                )


                backdoor_training_set, _ = load_data(
                    partition_id,
                    num_partitions,
                    alpha_val=0.9,
                    backdoor_enabled=True
                )

                net_clean = get_resnet_cnn_model()
                set_weights(net_clean, parameters)  # start from current global
                net_clean.to(self.device)

                clean_loss, clean_vec = train(
                    net_clean,
                    clean_training_set,
                    epochs=benign_epochs,  # or local_epochs, but you used 40 for attack
                    device=self.device,
                    lr=0.005  # benign-like LR (match your honest clients)
                )
                clean_delta = (clean_vec - init_vec.cpu()).detach().cpu()

                # 2) Build local "peer" reference clean deltas (bootstrapped)
                # Keep small for speed. Need at least krum_k+1.

                shared_seed = int(config.get("current-round", 0)) * 1000

                ref_deltas = build_reference_clean_deltas(
                    net=net_ref,  # base architecture (untrained clone is fine)
                    training_data=clean_training_set,
                    device=self.device,
                    init_vec=init_vec.cpu(),
                    epochs=benign_epochs,
                    lr=0.005,
                    num_refs=24,
                    seed_base=shared_seed,
                    label_smoothing=0.05,
                )

                # 3) Run constrained backdoor training using the BACKDOOR loader
                final_vec = train_constrain_and_scale_krum_proxy(
                    net=self.net,
                    training_data=backdoor_training_set,
                    device=self.device,
                    init_vec=init_vec.cpu(),
                    clean_delta=clean_delta,
                    ref_clean_deltas=ref_deltas,
                    krum_ref_delta=krum_ref_delta,
                    epochs=attack_epochs,  # you set local_epochs=40 for attacker
                    lr=0.005,  # you set 0.01 for attacker
                    label_smoothing=0.0,
                    weight_decay=0.0,

                    lambda_match_clean=1.0,
                    lambda_dir=0.2,
                    lambda_norm_match=0.2,
                    lambda_krum_proxy=1.0,

                    krum_k=7,  # if n=10 and f=1 => n-f-2=7, but proxy 3-5 is ok
                    min_norm_frac=0.10,
                )

                # 4) Krum-safe scaling: keep gamma SMALL
                # Base it on clean_delta norm, NOT prev_global step.
                attack_step = torch.norm(final_vec - init_vec.cpu()).item()
                clean_step = torch.norm(clean_delta).item()

                """
                gamma_cap = 2.0  # Krum safe cap, start conservative
                if clean_step > 1e-8 and attack_step > 1e-12:
                    gamma = min(gamma_cap, clean_step / attack_step)
                    gamma = max(1.0, gamma)
                else:
                    gamma = 1.0
                
                delta_adv = final_vec - init_vec.cpu()
                dist_to_clean = torch.norm(delta_adv - clean_delta).item()
                print(f"[Client {partition_id}][Round {current_round}] "
                      f"||Δ_adv||={delta_adv.norm().item():.4f} "
                      f"||Δ_clean||={clean_delta.norm().item():.4f} "
                      f"||Δ_adv-Δ_clean||={dist_to_clean:.4f} gamma={gamma:.2f}")
                dists = [torch.norm(delta_adv - d).item() for d in ref_deltas]
                dists.sort()
                print(f"[Client {partition_id}] Krum-proxy distances (7 smallest): {dists[:7]}")

                scaled_vec = krum_safe_scale(
                    final_vec=final_vec,
                    init_vec=init_vec.cpu(),
                    gamma=gamma,
                    keep_delta_norm=False,
                )
                """

                """
                vector_to_parameters(final_vec.to(self.device), self.net.parameters())
                self.prev_global_vec = init_vec.clone()

                return get_weights(self.net), len(backdoor_training_set.dataset), {
                    "attack": "constrain-and-scale-krum-proxy",
                    "clean_step": clean_step,
                    "attack_step": attack_step,
                }
                """

                delta_adv = final_vec - init_vec.cpu()

                # normalize direction
                delta_dir = delta_adv / (torch.norm(delta_adv) + 1e-12)

                # convert refs list -> tensor
                refs_tensor = torch.stack(ref_deltas)

                # compute benign update magnitude
                ref_norms = torch.norm(refs_tensor, dim=1)
                target_norm = ref_norms.median()

                #New target Norm
                target_norm = torch.norm(clean_delta)

                # enforce deterministic malicious update
                delta_adv = delta_dir * target_norm

                #New delta_adv with clean step norm
                delta_adv = clean_delta / torch.norm(clean_delta) * target_norm

                # reconstruct malicious weights
                final_vec = init_vec.cpu() + delta_adv

                vector_to_parameters(final_vec.to(self.device), self.net.parameters())

                self.prev_global_vec = init_vec.clone()

                return get_weights(self.net), len(backdoor_training_set.dataset), {
                    "attack": "constrain-and-scale-krum-proxy",
                    "clean_step": clean_step,
                    "attack_step": attack_step,
                }


        else:
            train_loss, final_vec = train(
                self.net,
                self.training_set,
                benign_epochs,
                self.device,
                0.005
            )

            self.prev_global_vec = init_vec.clone()

            return get_weights(self.net), len(self.training_set.dataset), {"train_loss": train_loss}

    def evaluate(self, parameters, config):
        partition_id = self.context.node_config["partition-id"]
        num_partitions = self.context.node_config["num-partitions"]
        _, self.test_set = load_data(partition_id, num_partitions, alpha_val=0.9)
        set_weights(self.net, parameters)
        loss, accuracy = test(self.net, self.test_set, self.device)
        # Evaluate Client Side Attack Success Rate
        client_side_asr = evaluate_asr(self.net, self.test_set, target_label=2, device=self.device)
        print(f"[Client {partition_id}] Completed evaluation: MTA={accuracy:.4f}, ASR={client_side_asr:.4f}")
        return loss, len(self.test_set.dataset), {"mta": accuracy, "asr": client_side_asr}

def client_fn(context: Context):

    # Load the default model
    #net = get_basic_cnn_model()
    #Load the updated model(ResNet)
    net = get_resnet_cnn_model()
    local_epochs = context.run_config["local-epochs"]
    partition_id = context.node_config["partition-id"]
    client = FlowerClient(net, local_epochs, context)

    #client.cid = str(partition_id)
    #print(f"Initialized client with partition ID: {partition_id} (CID set to {client.cid})")
    # Return Client instance
    return client.to_client()
    #return FlowerClient(net, local_epochs, context).to_client()

# Flower ClientApp
app = ClientApp(
    client_fn,
)
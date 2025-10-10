import os
import csv
import json
from datetime import datetime
from typing import Any, Dict, List, Optional

out_dir = "C:\\Users\\subra\\Research\\fed-learning-cifar-experiment"

def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def _append_csv(path: str, header: List[str], row: Dict[str, Any]):
    _ensure_dir(os.path.dirname(path) or ".")
    write_header = not os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        if write_header:
            writer.writeheader()
        writer.writerow({k: ("" if row.get(k) is None else row.get(k)) for k in header})

def write_experiment_summary(
    simulation_id: str,
    meta: Dict[str, Any],
    final_centralized_mta: float,
    final_centralized_asr: float,
    dist_mta_history: List[float],
    dist_asr_history: List[float],
    central_mta_history: List[float],
    central_asr_history: List[float],
    notes: Optional[str] = None,
):

    path = os.path.join(out_dir, "experiments.csv")
    header = [
        "simulation_id", "timestamp", "aggregation", "num_rounds", "local_epochs",
        "num_malicious_clients", "backdoor_attack_mode", "alpha",
        "final_centralized_mta", "final_centralized_asr",
        "dist_mta_history", "dist_asr_history",
        "central_mta_history", "central_asr_history",
        "notes"
    ]

    row = {
        "simulation_id": simulation_id,
        "timestamp": datetime.utcnow().isoformat(),
        "aggregation": meta.get("aggregation"),
        "num_rounds": meta.get("num_server_rounds"),
        "local_epochs": meta.get("local_epochs"),
        "num_malicious_clients": meta.get("num_malicious_clients"),
        "backdoor_attack_mode": meta.get("backdoor_attack_mode"),
        "alpha": meta.get("alpha"),
        "final_centralized_mta": f"{final_centralized_mta:.6f}" if final_centralized_mta is not None else "",
        "final_centralized_asr": f"{final_centralized_asr:.6f}" if final_centralized_asr is not None else "",
        "dist_mta_history": json.dumps(dist_mta_history),
        "dist_asr_history": json.dumps(dist_asr_history),
        "central_mta_history": json.dumps(central_mta_history),
        "central_asr_history": json.dumps(central_asr_history),
        "notes": notes or "",
    }
    _append_csv(path, header, row)

def append_centralized_round(simulation_id: str, rnd: int, centralized_loss: float,
                             centralized_mta: float, centralized_asr: float, num_clients: int):
    path = os.path.join(out_dir, "per_round_centralized.csv")
    header = ["simulation_id", "round", "timestamp", "centralized_loss", "centralized_mta", "centralized_asr", "num_clients"]
    row = {
        "simulation_id": simulation_id,
        "round": rnd,
        "timestamp": datetime.utcnow().isoformat(),
        "centralized_loss": centralized_loss,
        "centralized_mta": centralized_mta,
        "centralized_asr": centralized_asr,
        "num_clients": num_clients,
    }
    _append_csv(path, header, row)

def append_distributed_round(simulation_id: str, rnd: int, dist_mta: float, dist_asr: float, dist_loss: Optional[float], num_clients: int):
    path = os.path.join(out_dir, "per_round_distributed.csv")
    header = ["simulation_id", "round", "timestamp", "dist_loss", "dist_mta", "dist_asr", "num_clients"]
    row = {
        "simulation_id": simulation_id,
        "round": rnd,
        "timestamp": datetime.utcnow().isoformat(),
        "dist_loss": dist_loss,
        "dist_mta": dist_mta,
        "dist_asr": dist_asr,
        "num_clients": num_clients,
    }
    _append_csv(path, header, row)
#!/bin/bash
# =====================================================================
#  FL Backdoor Experiment — Neurotoxin / per-round-attack
#
#  Resource rationale
#  ------------------
#  10 clients sampled per round; each client actor gets 0.5 GPU.
#  4 GPUs → 8 concurrent client actors (Ray queues the remaining 2).
#  Bottleneck is the attacker: 40 epochs ≈ 60-90 s/round.
#  Server also runs on GPU (ref-delta computation, feature extraction).
#  100 rounds × ~4 min/round ≈ 7 h → request 10 h.
#
#  Submit:  sbatch fl_neurotoxin.slurm
# =====================================================================
#SBATCH --job-name=fl-neurotoxin
#SBATCH --account=V_robust_secure_AI_kislam4
#SBATCH --partition=defq
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=24          # 8 concurrent actors × 2 CPUs + server overhead
#SBATCH --mem=48G
#SBATCH --time=10:00:00
#SBATCH --output=logs/fl_neurotoxin_%j.out
#SBATCH --error=logs/fl_neurotoxin_%j.err

set -uo pipefail

# ---- Paths ----
PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$PROJECT_DIR"
mkdir -p logs

# ---- Conda ----
CONDA_BASE=$(conda info --base 2>/dev/null)
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate fed-learning-env

# ---- GPU visibility ----
# All 4 GPUs visible to Ray; client actors share them via fractional allocation.
export CUDA_VISIBLE_DEVICES=0,1,2,3

# Suppress Ray's per-message log deduplication so every client print appears.
export RAY_DEDUP_LOGS=0

# ---- Experiment parameters ----
# backdoor-partition-ids: the 10 fixed partitions that always attack when sampled.
# Pin these so runs are reproducible — copy the printed list from any prior run
# or keep these defaults.
SIMULATION_ID="neurotoxin-prnd-${SLURM_JOB_ID:-local}-$(date +%Y%m%d_%H%M%S)"
BACKDOOR_PARTITIONS='[5,12,23,36,44,51,67,78,83,91]'
NUM_ROUNDS=100
NUM_CLIENTS=100
LOCAL_EPOCHS=5
NUM_MALICIOUS_PER_ROUND=1
MASK_RATIO=0.01            # Neurotoxin: fraction of top-activity coords suppressed
SCALE_FACTOR=1.0           # No boosting — let structural stealth do the work
AGGREGATION=fedavg         # fedavg | krum | multikrum

# ---- Backend config ----
# 0.5 GPU per client actor → 8 clients run in parallel across 4 GPUs.
# Lower to 0.25 if OOM; raise to 1.0 to fully isolate each actor (slower).
BACKEND_CONFIG='{"client-resources": {"num-cpus": 2, "num-gpus": 0.5}}'

# =====================================================================
echo "========================================================"
echo "  FL Neurotoxin Backdoor Experiment"
echo "  Job ID   : ${SLURM_JOB_ID:-interactive}"
echo "  Node     : $(hostname)"
echo "  GPUs     : $(nvidia-smi --query-gpu=name --format=csv,noheader \
                     2>/dev/null | tr '\n' ',' || echo 'N/A')"
echo "  Start    : $(date)"
echo "--------------------------------------------------------"
echo "  Sim ID   : $SIMULATION_ID"
echo "  Rounds   : $NUM_ROUNDS"
echo "  Attack   : per-round-attack / neurotoxin"
echo "  Backdoor : $BACKDOOR_PARTITIONS"
echo "  Aggr     : $AGGREGATION"
echo "========================================================"
echo ""

# =====================================================================
#  Main run
#  We call flower-simulation directly (what `flwr run` wraps internally)
#  to override the federation's num-gpus=0 default without touching
#  pyproject.toml.
# =====================================================================
flower-simulation \
    --app . \
    --num-supernodes "$NUM_CLIENTS" \
    --backend-config "$BACKEND_CONFIG" \
    --run-config \
        "num-server-rounds=$NUM_ROUNDS \
         num-clients=$NUM_CLIENTS \
         local-epochs=$LOCAL_EPOCHS \
         fraction-fit=0.1 \
         backdoor-attack-mode='per-round-attack' \
         backdoor-attack-type='neurotoxin' \
         num-malicious-clients-per-round=$NUM_MALICIOUS_PER_ROUND \
         num-malicious-clients=$NUM_MALICIOUS_PER_ROUND \
         aggregation-method='$AGGREGATION' \
         attacker-selection-mode='random' \
         simulation-id='$SIMULATION_ID' \
         backdoor-partition-ids='$BACKDOOR_PARTITIONS' \
         neurotoxin-mask-ratio=$MASK_RATIO \
         neurotoxin-scale-factor=$SCALE_FACTOR"

EXIT_CODE=$?

echo ""
echo "========================================================"
echo "  Job finished : $(date)"
echo "  Exit code    : $EXIT_CODE"
if [ $EXIT_CODE -eq 0 ]; then
    echo "  Results in   : per_update_features.csv"
fi
echo "========================================================"
exit $EXIT_CODE

#!/bin/bash
#SBATCH --job-name=valuehead_lineex
#SBATCH --partition=gpu_a100_short
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64gb
#SBATCH --time=00:30:00
#SBATCH --output=logs/valuehead_%j.out
#SBATCH --error=logs/valuehead_%j.err
#SBATCH --export=NONE

set -euo pipefail
mkdir -p logs

# Minimal environment (weil export=NONE)
export PATH=/usr/bin:/bin
export HOME="$HOME"
export SLURM_SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"

unset CONDA_DEFAULT_ENV CONDA_PREFIX CONDA_PROMPT_MODIFIER CONDA_PYTHON_EXE CONDA_SHLVL
unset _CE_CONDA _CE_M

module purge
module load devel/cuda/11.8
module load devel/miniforge

CONDA_EXE="/opt/bwhpc/common/devel/miniforge/25.3.1-py3.12/bin/conda"

# Headless safe
export QT_QPA_PLATFORM=offscreen
export MPLBACKEND=Agg
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd "$SLURM_SUBMIT_DIR"

echo "HOST: $(hostname)"
echo "DATE: $(date)"
echo "PWD:  $(pwd)"
echo "SLURM_JOB_ID: ${SLURM_JOB_ID:-}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-}"

nvidia-smi || true

# -------- PATHS --------
JEPA_DIR="/pfs/data6/home/ul/ul_student/ul_upf92/Project/jepa"

# Daten aus Generator:
DATA_DIR="$JEPA_DIR/out_lineex"
MANIFEST="$DATA_DIR/manifest.jsonl"

# JEPA Checkpoints:
JEPA_CKPT_DIR="$JEPA_DIR/outputs/jepa_lineex"

# Output vom ValueHead:
VH_OUTDIR="$JEPA_DIR/outputs/valuehead_lineex"
mkdir -p "$VH_OUTDIR"
# -----------------------

# Optional: WANDB Settings (falls du willst)
export WANDB_PROJECT="lineex-project"
export WANDB_RUN_NAME="valuehead_K64_run1"
# export WANDB_ENTITY="dein_username_oder_team"   # nur wenn nötig
# export WANDB_MODE="online"                      # oder "offline"

# --- Pick checkpoint automatically if not set ---
CHECKPOINT="${CHECKPOINT:-}"
if [[ -z "${CHECKPOINT}" ]]; then
  # bevorzugt: latest, sonst neuester *.tar
  if ls "$JEPA_CKPT_DIR"/*latest*.pth.tar >/dev/null 2>&1; then
    CHECKPOINT="$(ls -1t "$JEPA_CKPT_DIR"/*latest*.pth.tar | head -n 1)"
  elif ls "$JEPA_CKPT_DIR"/*.pth.tar >/dev/null 2>&1; then
    CHECKPOINT="$(ls -1t "$JEPA_CKPT_DIR"/*.pth.tar | head -n 1)"
  elif ls "$JEPA_CKPT_DIR"/*.tar >/dev/null 2>&1; then
    CHECKPOINT="$(ls -1t "$JEPA_CKPT_DIR"/*.tar | head -n 1)"
  else
    echo "ERROR: No checkpoint found in $JEPA_CKPT_DIR"
    exit 1
  fi
fi

echo "DATA_DIR:     $DATA_DIR"
echo "MANIFEST:     $MANIFEST"
echo "JEPA_CKPT:    $CHECKPOINT"
echo "VH_OUTDIR:    $VH_OUTDIR"

# Sanity checks
test -f "$MANIFEST" || { echo "ERROR: manifest not found: $MANIFEST"; exit 1; }
test -d "$DATA_DIR/images" || { echo "ERROR: images folder missing: $DATA_DIR/images"; exit 1; }
test -d "$DATA_DIR/labels" || { echo "ERROR: labels folder missing: $DATA_DIR/labels"; exit 1; }
test -f "$CHECKPOINT" || { echo "ERROR: checkpoint not found: $CHECKPOINT"; exit 1; }

cd "$JEPA_DIR"

# Torch sanity
"$CONDA_EXE" run -n jepa --no-capture-output python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.device_count())"

# Run ValueHead training
"$CONDA_EXE" run -n jepa --no-capture-output \
  python -u train_valuehead.py \
    --out_dir "$VH_OUTDIR" \
    --data_dir "$DATA_DIR" \
    --manifest "$MANIFEST" \
    --jepa_ckpt "$CHECKPOINT" \
    --K 64 \
    --epochs 20 \
    --batch_size 64 \
    --lr 1e-3 \
    --fp16 \
    --use_wandb \
    --wandb_project "$WANDB_PROJECT" \
    --wandb_run_name "$WANDB_RUN_NAME"
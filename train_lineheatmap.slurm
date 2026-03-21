#!/bin/bash
#SBATCH --job-name=lineheatmap_lineex
#SBATCH --partition=gpu_a100_short
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64gb
#SBATCH --time=00:30:00
#SBATCH --output=logs/lineheatmap_%j.out
#SBATCH --error=logs/lineheatmap_%j.err
#SBATCH --export=NONE

set -euo pipefail
mkdir -p logs

export PATH=/usr/bin:/bin
export HOME="$HOME"
export SLURM_SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"

unset CONDA_DEFAULT_ENV CONDA_PREFIX CONDA_PROMPT_MODIFIER CONDA_PYTHON_EXE CONDA_SHLVL
unset _CE_CONDA _CE_M

module purge
module load devel/cuda/11.8
module load devel/miniforge

CONDA_EXE="/opt/bwhpc/common/devel/miniforge/25.3.1-py3.12/bin/conda"

export QT_QPA_PLATFORM=offscreen
export MPLBACKEND=Agg
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export WANDB_MODE=online

JEPA_DIR="/pfs/data6/home/ul/ul_student/ul_upf92/Project/jepa"
CONFIG="$JEPA_DIR/configs/lineheatmap/lineheatmap_vitl16v2.yaml"

cd "$SLURM_SUBMIT_DIR"

echo "HOST: $(hostname)"
echo "DATE: $(date)"
echo "PWD:  $(pwd)"
echo "SLURM_JOB_ID: ${SLURM_JOB_ID:-}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-}"

nvidia-smi || true

test -f "$CONFIG" || { echo "ERROR: config not found: $CONFIG"; exit 1; }

cd "$JEPA_DIR"

echo "=== CONFIG CHECK ==="
grep -n "encoder_checkpoint" "$CONFIG" || true
grep -n "train_manifest" "$CONFIG" || true
grep -n "val_manifest" "$CONFIG" || true

"$CONDA_EXE" run -n jepa --no-capture-output python -c \
"import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.device_count())"

"$CONDA_EXE" run -n jepa --no-capture-output \
  python -u -m app.lineheatmap2.train --config "$CONFIG"
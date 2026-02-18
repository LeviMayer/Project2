#!/bin/bash
#SBATCH --job-name=jepa_lineex
#SBATCH --partition=gpu_a100_il
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64gb
#SBATCH --time=04:00:00
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err
#SBATCH --export=NONE

set -euo pipefail
mkdir -p logs

# Minimal sane environment (weil export=NONE)
export PATH=/usr/bin:/bin
export HOME="$HOME"
export SLURM_SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"

echo "HOST: $(hostname)"
echo "DATE: $(date)"
echo "PWD:  $(pwd)"
echo "SLURM_JOB_ID: ${SLURM_JOB_ID:-}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-}"

# Remove inherited conda markers (extra safety)
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

nvidia-smi || true

# --- TODO: set these paths ---
JEPA_DIR="/pfs/data6/home/ul/ul_student/ul_upf92/Project/jepa"
OUTDIR="/pfs/data6/home/ul/ul_student/ul_upf92/Project/jepa/outputs/jepa_lineex"
CONFIG="$JEPA_DIR/configs/pretrain/lineex_vitl16.yaml"
# -----------------------------

mkdir -p "$OUTDIR"

# Sanity check
"$CONDA_EXE" run -n jepa --no-capture-output python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.device_count())"

# Run JEPA
cd "$JEPA_DIR"
"$CONDA_EXE" run -n jepa --no-capture-output \
  python -u -m app.main --fname "$CONFIG"
 



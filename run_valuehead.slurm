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

# minimal environment
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

cd "$SLURM_SUBMIT_DIR"

echo "HOST: $(hostname)"
echo "DATE: $(date)"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-}"

nvidia-smi || true

# -------- PATHS --------
JEPA_DIR="/pfs/data6/home/ul/ul_student/ul_upf92/Project/jepa"
OUTDIR="$JEPA_DIR/outputs/jepa_lineex"
MANIFEST="$OUTDIR/manifest.jsonl"
CHECKPOINT=""   # optional: z.B. "$OUTDIR/lineex_vjepa-latest.pth.tar"
# -----------------------

cd "$JEPA_DIR"

"$CONDA_EXE" run -n jepa --no-capture-output \
  python -u train_valuehead.py \
    --out_dir "$OUTDIR" \
    --manifest "$MANIFEST" \
    --K 64 \
    --epochs 20 \
    --batch_size 64 \
    --lr 1e-3 \
    --fp16 \
    --use_wandb \
    --wandb_project lineex-project \
    --wandb_run_name valuehead_K64_run1
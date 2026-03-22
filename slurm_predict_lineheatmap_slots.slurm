#!/bin/bash
#SBATCH --job-name=predict_slots
#SBATCH --partition=gpu_a100_short
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32gb
#SBATCH --time=00:30:00
#SBATCH --output=logs/predict_slots_%j.out
#SBATCH --error=logs/predict_slots_%j.err
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
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

JEPA_DIR="/pfs/data6/home/ul/ul_student/ul_upf92/Project/jepa"
CONFIG="$JEPA_DIR/configs/lineheatmap/lineheatmap_vitl16v2.yaml"
CHECKPOINT="$JEPA_DIR/outputs/lineheatmap_vitl16_slots/best.pth"
OUT_DIR="$JEPA_DIR/outputs/predictions_slots_v3"

cd "$JEPA_DIR"

echo "HOST: $(hostname)"
echo "DATE: $(date)"
echo "PWD:  $(pwd)"
echo "SLURM_JOB_ID: ${SLURM_JOB_ID:-}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-}"

nvidia-smi || true

test -f "$CONFIG" || { echo "ERROR: config not found: $CONFIG"; exit 1; }
test -f "$CHECKPOINT" || { echo "ERROR: checkpoint not found: $CHECKPOINT"; exit 1; }

mkdir -p "$OUT_DIR"

"$CONDA_EXE" run -n jepa --no-capture-output \
  python -u -m app.lineheatmap2.predict_heatmap \
    --config "$CONFIG" \
    --checkpoint "$CHECKPOINT" \
    --num_samples 200 \
    --start_index 0 \
    --out_dir "$OUT_DIR"
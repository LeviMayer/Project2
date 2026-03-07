#!/bin/bash
#SBATCH --job-name=gen_lineex
#SBATCH --partition=dev_cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=16gb
#SBATCH --time=00:30:00
#SBATCH --output=logs/gen_%j.out
#SBATCH --error=logs/gen_%j.err
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

export MPLBACKEND=Agg
export QT_QPA_PLATFORM=offscreen

cd "$SLURM_SUBMIT_DIR"

JEPA_DIR="/pfs/data6/home/ul/ul_student/ul_upf92/Project/jepa"

cd "$JEPA_DIR"

echo "Generating synthetic charts..."
echo "Host: $(hostname)"
echo "Date: $(date)"

"$CONDA_EXE" run -n jepa --no-capture-output \
  python gen_linecharts.py

echo "Done."
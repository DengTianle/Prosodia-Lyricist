#!/bin/bash
#SBATCH --job-name=prosodia-end-to-end
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100-80:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:59:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(dirname "${BASH_SOURCE[0]}")}"

# Use the environment's interpreter without depending on the base Conda launcher.
PROSODIA_ENV="${PROSODIA_ENV:-$HOME/miniconda3/envs/prosodia-lyricist}"
if [[ ! -x "$PROSODIA_ENV/bin/python" ]]; then
    echo "Python not found at $PROSODIA_ENV/bin/python; set PROSODIA_ENV to the environment path." >&2
    exit 1
fi
export PATH="$PROSODIA_ENV/bin:$PATH"

time "$PROSODIA_ENV/bin/python" -m prosodia_lyricist.train --config "${1:-configs/end_to_end.yaml}"

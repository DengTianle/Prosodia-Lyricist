#!/bin/bash
#SBATCH --time=02:59:00
#SBATCH --mem=32G
#SBATCH --output=logs/prep-bridge-%j.out
#SBATCH --error=logs/prep-bridge-%j.err

source ~/miniconda3/etc/profile.d/conda.sh
conda activate prosodia-lyricist

export PATH_ESPEAK="$(python -c 'import espeak_english; print(espeak_english.library_path())')"
export PHONEMIZER_ESPEAK_LIBRARY="$PATH_ESPEAK"

export ESPEAK_DATA_PATH="$(python -c 'import espeak_english; print(espeak_english.data_path())')"
export PHONEMIZER_ESPEAK_DATA_PATH="$ESPEAK_DATA_PATH"

srun python -m prosodia_lyricist.prepare --config configs/bridge.yaml

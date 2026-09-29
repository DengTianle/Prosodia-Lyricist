#!/bin/bash
#SBATCH --job-name=prosodia-bridge
#SBATCH --partition=gpu-long
#SBATCH --gres=gpu:
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

source ~/miniconda3/etc/profile.d/conda.sh
conda activate prosodia-lyricist

time srun prosodia-train --config configs/bridge.yaml

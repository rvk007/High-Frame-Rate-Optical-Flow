#!/bin/bash
#SBATCH --time=4:00:00
#SBATCH --mem=15GB
#SBATCH --job-name=combined
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:rtx8000:1

module purge
module load anaconda3/2020.07
eval "$(conda shell.bash hook)"
conda activate craft

blenderproc run examples/advanced/optical_flow/main.py examples/advanced/optical_flow/camera_positions examples/resources/combined.obj examples/advanced/optical_flow/output 3

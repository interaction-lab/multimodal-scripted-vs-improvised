#!/bin/bash
#SBATCH --export=NONE
#SBATCH --job-name=dac_all_models
#SBATCH --account=dilkina_1463
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100|v100"
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=48:00:00
#SBATCH --nodes=1

#SBATCH --output=/home1/anurades/multimodal-scripted-vs-improvised/logs/dac_all_%j.out
#SBATCH --error=/home1/anurades/multimodal-scripted-vs-improvised/logs/dac_all_%j.err

# -------------------------
# 1. Load modules
# -------------------------

module load conda
eval "$(conda shell.bash hook)"
conda activate multimodal

cd /home1/anurades/multimodal-scripted-vs-improvised/multimodal-dac


# -------------------------
# 2. Run the experiment
# -------------------------
python video.py

echo "Job completed."
echo "End time: $(date)"

#!/bin/bash
#SBATCH --job-name=demo_wp_4
#SBATCH --output=/work/pnrr_itserr/WP8-embeddings/logs/%x-%j
#SBATCH --error=/work/pnrr_itserr/WP8-embeddings/logs/%x-%j
#SBATCH --open-mode=truncate
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=8
#SBATCH --account=pnrr_itserr
#SBATCH --time=24:00:00
#SBATCH --partition=all_usr_prod

source /etc/profile.d/modules.sh
module unload cuda
module load cuda/11.8

source activate itserr
cd /homes/fcocchi/Latin-Document-Search-Engine

export PYTHONPATH="."
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1

srun python ui_launcher.py

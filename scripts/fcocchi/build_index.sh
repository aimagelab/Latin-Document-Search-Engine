#!/bin/bash
#SBATCH --job-name=build_index
#SBATCH --output=/work/pnrr_itserr/WP8-embeddings/logs/folder_structure/%x-%j
#SBATCH --error=/work/pnrr_itserr/WP8-embeddings/logs/folder_structure/%x-%j
#SBATCH --open-mode=truncate
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --mem=256G
#SBATCH --cpus-per-task=8
#SBATCH --account=pnrr_itserr
#SBATCH --time=06:00:00
#SBATCH --partition=all_usr_prod
##SBATCH --constraint="gpu_A40_48G|gpu_RTX5000_16G|gpu_RTXA5000_24G|gpu_RTX6000_24G|gpu_2080Ti_11G"

source /etc/profile.d/modules.sh
module unload cuda
module load cuda/11.8

source activate itserr
cd /homes/fcocchi/Latin-Document-Search-Engine

export PYTHONPATH="."
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1

srun python preprocessing_data/build_index.py

#!/bin/bash
#SBATCH --job-name=write_preds_save_dice
#SBATCH --cpus-per-gpu=6
#SBATCH --mem=60G
#SBATCH --partition=gpu-long
#SBATCH --gres=gpu:1
#SBATCH --time=160:00:00
#SBATCH -e logs/slurm-%j.err
#SBATCH -o logs/slurm-%j.out
#SBATCH --mail-type=end
#SBATCH --mail-user=jiajingnan2222@gmail.com

##SBATCH --nodelist=node860


echo "on Hostname = $(hostname)"
echo "on GPU      = $CUDA_VISIBLE_DEVICES"
echo
echo "@ $(date)"
echo

eval $(conda shell.bash hook)

conda activate py37
stdbuf -oL python -u write_preds_save_dice.py







#!/bin/bash 
#SBATCH --job-name=vessel
##SBATCH --output=output_vessel_only_medium.txt
#SBATCH --ntasks=1 
#SBATCH --cpus-per-task=4

#SBATCH --mem=60G

#SBATCH --partition=gpu-long

#SBATCH --nodelist=node852

#SBATCH --gres=gpu:1
#SBATCH --time=160:00:00 

echo "on Hostname = $(hostname)"
echo "on GPU      = $CUDA_VISIBLE_DEVICES"
echo
echo "@ $(date)"
echo

eval $(conda shell.bash hook)

conda activate py37
python train_ori_fit_rec_epoch.py 



#!/bin/bash 
#SBATCH --job-name=vessel
##SBATCH --output=output_vessel_only_medium.txt
#SBATCH --ntasks=1 
#SBATCH --cpus-per-task=1

#SBATCH --mem=20G

#SBATCH --partition=gpu-long

##SBATCH --nodelist=node860

#SBATCH --gres=gpu:1
#SBATCH --time=160:00:00 
#SBATCH -e logs/slurm-%j.err
#SBATCH -o logs/slurm-%j.out

echo "on Hostname = $(hostname)"
echo "on GPU      = $CUDA_VISIBLE_DEVICES"
echo
echo "@ $(date)"
echo

eval $(conda shell.bash hook)

conda activate py37
##stdbuf -oL python -u write_preds_save_dice.py
stdbuf -oL python -u train_ori_fit_rec_epoch.py --mtscale=1 --p_middle=0.5 --tr_nb=18 --no_label_nb=0 --patches_per_scan=100 --model_names='net_only_lobe' --no_label_dir='None' --lr=0.0001 --load=1
##stdbuf -oL python write_preds_save_dice.py
##python plot_curve.py






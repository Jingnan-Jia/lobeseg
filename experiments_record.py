


# those following code is for experiments  for adaptive_lr=1, node=node855, 2020-09-19-03-24

#!/bin/bash
#SBATCH --partition=gpu-long
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-gpu=6
#SBATCH -t 7-00:00:00
##SBATCH --mem-per-gpu=70G
#SBATCH --mail-type=end
#SBATCH --mail-user=jiajingnan2222@gmail.com

eval $(conda shell.bash hook)
conda activate py37

current_time=$(date +%y_%m_%d_%H_%M_%S)  # there must not be any space before and after =

export CUDA_VISIBLE_DEVICES=0
stdbuf -oL python -u train_ori_fit_rec_epoch.py 2>logs/slurm-${current_time}_1.err 1>logs/slurm-${current_time}_1.out --mtscale=0 --p_middle=0.0 --step_nb=100001 --adaptive_lr=1 --model_names='net_only_lobe' --lr_lb=0.0001 &
export CUDA_VISIBLE_DEVICES=1
stdbuf -oL python -u train_ori_fit_rec_epoch.py 2>logs/slurm-${current_time}_2.err 1>logs/slurm-${current_time}_2.out --mtscale=0 --p_middle=0.0 --step_nb=50001 --adaptive_lr=1 --model_names='net_only_lobe-net_no_label' --lr_lb=0.0001 &
export CUDA_VISIBLE_DEVICES=2
stdbuf -oL python -u train_ori_fit_rec_epoch.py 2>logs/slurm-${current_time}_3.err 1>logs/slurm-${current_time}_3.out --mtscale=0 --p_middle=0.0 --step_nb=50001 --adaptive_lr=1 --model_names='net_only_lobe-net_only_vessel' --lr_lb=0.0001 &
export CUDA_VISIBLE_DEVICES=3
stdbuf -oL python -u train_ori_fit_rec_epoch.py 2>logs/slurm-${current_time}_4.err 1>logs/slurm-${current_time}_4.out --mtscale=0 --p_middle=0.0 --step_nb=50001 --adaptive_lr=1 --model_names='net_only_lobe-net_no_label-net_only_vessel' --lr_lb=0.0001 &

wait




#!/bin/bash
#SBATCH --partition=gpu-long
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-gpu=8
#SBATCH -t 7-00:00:00
#SBATCH --mem-per-gpu=150G
#SBATCH --mail-type=end
#SBATCH --mail-user=jiajingnan2222@gmail.com

eval $(conda shell.bash hook)
conda activate py37

current_time=$(date +%y_%m_%d_%H_%M_%S)  # there must not be any space before and after =

export CUDA_VISIBLE_DEVICES=0
stdbuf -oL python -u train_ori_fit_rec_epoch.py 2>logs/slurm-${current_time}_0.err 1>logs/slurm-${current_time}_0.out --p_middle=0.5 --step_nb=100001 --adaptive_lr=0 --fat=0 --model_names='net_only_lobe-net_no_label-net_only_vessel' --lb_io="1_in_low_1_out_low" --rc_io="1_in_hgh_1_out_hgh" --vs_io="1_in_hgh_1_out_hgh" &
export CUDA_VISIBLE_DEVICES=1
stdbuf -oL python -u train_ori_fit_rec_epoch.py 2>logs/slurm-${current_time}_1.err 1>logs/slurm-${current_time}_1.out --p_middle=0.5 --step_nb=100001 --adaptive_lr=1 --fat=0 --model_names='net_only_lobe-net_no_label-net_only_vessel' --lb_io="1_in_low_1_out_low" --rc_io="1_in_hgh_1_out_hgh" --vs_io="1_in_hgh_1_out_hgh" &

wait

#export CUDA_VISIBLE_DEVICES=0
#stdbuf -oL python -u train_ori_fit_rec_epoch.py 2>logs/slurm-${current_time}_2.err 1>logs/slurm-${current_time}_2.out --mtscale=0 --p_middle=0.5 --step_nb=50001 --adaptive_lr=1 --model_names='net_only_lobe-net_no_label' &
#export CUDA_VISIBLE_DEVICES=1
#stdbuf -oL python -u train_ori_fit_rec_epoch.py 2>logs/slurm-${current_time}_3.err 1>logs/slurm-${current_time}_3.out --mtscale=1 --p_middle=0.5 --step_nb=50001 --adaptive_lr=0 --model_names='net_only_lobe-net_no_label-net_only_vessel' -ld_lb_name='1600479252_877_lrlb0.0001lrvs1e-05mtscale1netnol-nnl-novpm0.0nldLUNA16ao0ds0pps100lbnb17vsnb50nlnb400ptsz144ptzsz96' --ld_rc_name='1600479252_840_lrlb0.0001lrvs1e-05mtscale1netnol-nnl-novpm0.0nldLUNA16ao0ds0pps100lbnb17vsnb50nlnb400ptsz144ptzsz96' --ld_vs_name='1600479252_659_lrlb0.0001lrvs1e-05mtscale1netnol-nnl-novpm0.0nldLUNA16ao0ds0pps100lbnb17vsnb50nlnb400ptsz144ptzsz96' &
#
#wait

#export CUDA_VISIBLE_DEVICES=0
#stdbuf -oL python -u train_ori_fit_rec_epoch.py 2>logs/slurm-${current_time}_1.err 1>logs/slurm-${current_time}_1.out --mtscale=0 --p_middle=0 --step_nb=100001 --adaptive_lr=0 --model_names='net_only_lobe' --ld_lb_name='1600645872_556_lrlb0.0001lrvs1e-05mtscale0netnolpm0.0nldLUNA16ao0ds0pps100lbnb17vsnb50nlnb400ptsz144ptzsz96' &
#export CUDA_VISIBLE_DEVICES=1
#stdbuf -oL python -u train_ori_fit_rec_epoch.py 2>logs/slurm-${current_time}_2.err 1>logs/slurm-${current_time}_2.out --mtscale=0 --p_middle=0.5 --step_nb=50001 --adaptive_lr=0 --model_names='net_only_lobe-net_no_label' --ld_lb_name='1600645872_657_lrlb0.0001lrvs1e-05mtscale0netnol-nnlpm0.5nldLUNA16ao0ds0pps100lbnb17vsnb50nlnb400ptsz144ptzsz96' --ld_rc_name='1600645872_320_lrlb0.0001lrvs1e-05mtscale0netnol-nnlpm0.5nldLUNA16ao0ds0pps100lbnb17vsnb50nlnb400ptsz144ptzsz96' &
#export CUDA_VISIBLE_DEVICES=2
#stdbuf -oL python -u train_ori_fit_rec_epoch.py 2>logs/slurm-${current_time}_3.err 1>logs/slurm-${current_time}_3.out --mtscale=0 --p_middle=0.5 --step_nb=50001 --adaptive_lr=0 --model_names='net_only_lobe-net_only_vessel' &
#export CUDA_VISIBLE_DEVICES=3
#stdbuf -oL python -u train_ori_fit_rec_epoch.py 2>logs/slurm-${current_time}_4.err 1>logs/slurm-${current_time}_4.out --mtscale=0 --p_middle=0.5 --step_nb=50001 --adaptive_lr=0 --model_names='net_only_lobe-net_no_label-net_only_vessel' --ld_lb_name='1600645872_801_lrlb0.0001lrvs1e-05mtscale0netnol-nnl-novpm0.5nldLUNA16ao0ds0pps100lbnb17vsnb50nlnb400ptsz144ptzsz96' --ld_rc_name='1600645872_404_lrlb0.0001lrvs1e-05mtscale0netnol-nnl-novpm0.5nldLUNA16ao0ds0pps100lbnb17vsnb50nlnb400ptsz144ptzsz96' --ld_vs_name='1600645872_282_lrlb0.0001lrvs1e-05mtscale0netnol-nnl-novpm0.5nldLUNA16ao0ds0pps100lbnb17vsnb50nlnb400ptsz144ptzsz96' &
#
#wait




#!/bin/bash 
#SBATCH --job-name=single_train
#SBATCH --cpus-per-gpu=6
#SBATCH --mem=60G
#SBATCH --partition=gpu-long
#SBATCH --gres=gpu:1
#SBATCH --time=160:00:00
#SBATCH --mail-type=end
#SBATCH --mail-user=jiajingnan2222@gmail.com

##SBATCH -e logs/slurm-%j.err
##SBATCH -o logs/slurm-%j.out
##SBATCH --output=output_vessel_only_medium.txt
##SBATCH --nodelist=node860
##SBATCH --ntasks=1

echo "on Hostname = $(hostname)"
echo "on GPU      = $CUDA_VISIBLE_DEVICES"
echo
echo "@ $(date)"
echo

eval $(conda shell.bash hook)

conda activate py37
stdbuf -oL python -u train_ori_fit_rec_epoch.py>logs/slurm-000305002.out 2>&1 --mtscale=1 --p_middle=0 --step_nb=50001 --lb_tr_nb=18 --patches_per_scan=100 --trgt_space=1.4 --trgt_z_space=2.5 --model_names='net_only_lobe' --lr_lb=0.0001 --ld_lb_name='1598736397_998_lrlb0.0001lrvs1e-05mtscale1netnol-novpm0.5nldLUNA16ao0ds0tsp1.4z2.5pps100lbnb18vsnb50nlnb400ptsz144ptzsz96'







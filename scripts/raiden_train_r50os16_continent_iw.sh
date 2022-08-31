#!/bin/bash
#$ -S /bin/bash
#$ -cwd
#$ -j y
#$ -jc gpu-container_g8.24h
#$ -ac d=nvcr-pytorch-2010,d_shm=128G
#$ -N oemiw5k
. /home/songjian/net.sh
# export CUDA_VISIBLE_DEVICES=0, 1, 2, 3

domains=(asia africa europe centralamerica northamerica southamerica oceania)
src=${domains[$(($SGE_TASK_ID-1))]}
unset domains[$(($SGE_TASK_ID-1))]

/home/songjian/.conda/envs/robustnet/bin/python -m torch.distributed.launch --nproc_per_node=8 train.py \
        --dataset $src \
        --covstat_val_dataset $src \
        --val_dataset ${domains[*]} \
        --arch network.deepv3.DeepR50V3PlusD \
        --city_mode 'train' \
        --lr_schedule poly \
        --lr 0.01 \
        --poly_exp 0.9 \
        --max_cu_epoch 10000 \
        --class_uniform_pct 0.5 \
        --class_uniform_tile 1024 \
        --crop_size 512 \
        --scale_min 0.5 \
        --scale_max 2.0 \
        --rrotate 0 \
        --max_iter 5000 \
        --bs_mult 8 \
        --gblur \
        --color_aug 0.5 \
        --wt_reg_weight 0.8 \
        --relax_denom 0.0 \
        --clusters 0 \
        --cov_stat_epoch 0 \
        --trials 0 \
        --wt_layer 0 0 1 1 1 0 0 \
        --date 0731v100_$src \
        --exp r50os16_oem_iw_bs8_g8_5k_$src \
        --ckpt ./logs/ \
        --tb_path ./logs/

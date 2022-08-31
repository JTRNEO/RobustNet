#!/bin/bash
#$ -S /bin/bash
#$ -cwd
#$ -j y
#$ -jc gtn-container_g8.24h
#$ -ac d=nvcr-pytorch-2010,d_shm=128G
#$ -N isw
. /home/songjian/net.sh
# export CUDA_VISIBLE_DEVICES=0, 1, 2, 3
/home/songjian/.conda/envs/robustnet/bin/python -m torch.distributed.launch --nproc_per_node=8 train.py \
        --dataset gtav \
        --covstat_val_dataset gtav \
        --val_dataset bdd100k cityscapes synthia mapillary \
        --arch network.deepv3.DeepR50V3PlusD \
        --city_mode 'train' \
        --lr_schedule poly \
        --lr 0.01 \
        --poly_exp 0.9 \
        --max_cu_epoch 10000 \
        --class_uniform_pct 0.5 \
        --class_uniform_tile 1024 \
        --crop_size 768 \
        --scale_min 0.5 \
        --scale_max 2.0 \
        --rrotate 0 \
        --max_iter 5000 \
        --bs_mult 16 \
        --gblur \
        --color_aug 0.5 \
        --wt_reg_weight 0.6 \
        --relax_denom 0.0 \
        --clusters 3 \
        --cov_stat_epoch 5 \
        --trials 10 \
        --wt_layer 0 0 2 2 2 0 0 \
        --date 0627SV \
        --exp r50os16_gtav_isw_bs16_g8 \
        --ckpt ./logs/ \
        --tb_path ./logs/

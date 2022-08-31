#!/bin/bash
domains=(asia africa europe centralamerica northamerica southamerica oceania)
src=${domains[0]}
unset domains[0]
echo $src
echo ${domains[*]}
/home/songjian/anaconda3/envs/robustnet/bin/python -m torch.distributed.launch --nproc_per_node=1 train.py \
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
        --max_iter 1000 \
        --bs_mult 8 \
        --gblur \
        --color_aug 0.5 \
        --wt_reg_weight 0.0 \
        --relax_denom 0.0 \
        --cov_stat_epoch 0 \
        --wt_layer 0 0 -1 -1 -1 0 0 \
        --date 0729$src \
        --exp r50os16_base_bs8_g8_5k_$src \
        --ckpt ./logs_debug/ \
        --tb_path ./logs_debug/ \
        --test_mode

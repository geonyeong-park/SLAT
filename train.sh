#!/bin/bash

<< "END"
advcoeff=(0.1 0.05 0.01 0.005)
DGlr=(0.0003 0.0005)
pixAdv='LS'
Gfake_cyc=(0.1 1)
END

pgditer=0
if [ "$2" == "PGD" ]; then
    pgditer=$3
fi
eps_list=( 9 10 )

GA_list=( 1. )
alpha_list=( 0.5 0.6 )

for GA in ${GA_list[@]}; do
    for alpha in ${alpha_list[@]}; do
        expname="CIFAR10_cycle_eps8_GA"${GA}"_alpha"${alpha}""

        CUDA_VISIBLE_DEVICES=1 python3 main.py --gpu 1 --dataset_name 'CIFAR10' --model_structure advGNI_GA \
            --exp_name "$expname" --eta 8 --GA_coeff $GA --alpha $alpha
    done
done

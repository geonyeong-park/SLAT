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
eps_list=( 8 )

for eps in ${eps_list[@]}; do
    for seed in {2..4}; do
        if [ "$2" == "PGD" ]; then
            expname="CIFAR10_eps"$eps"_"$2""$pgditer"_alpha_seed"${seed}""
        else
            expname="CIFAR10_eps"$eps"_"$2"_seed"${seed}""
        fi

        CUDA_VISIBLE_DEVICES=$1 python3 main.py --gpu $1 --dataset_name 'CIFAR10' --model_structure $2 \
            --exp_name "$expname" --eta $eps --PGD_iters $pgditer \
            --resume "snapshots/"$expname"/pretrain.pth" 
        done
done

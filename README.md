# Reliably fast adversarial training via latent adversarial perturbation
Pytorch implementation of SLAT: single-step latent adversarial training method.
Provided as a supplementary code for ICCV 2021. 
- Pytorch version: 1.7.1
- Autoattack (croce et al., 2020) should be installed. (https://github.com/fra31/auto-attack)

## Train examples

```
python3 main.py --gpu 0 --model_structure advGNI --eta 8 --exp_name CIFAR10_test --dataset_name CIFAR10 

```
- model_structure: Other baselines (FGSM, FGSM-RS, FGSM-GA, PGD, etc) can be compared. Please refer to the config.yaml
- dataset_name: CIFAR-10, CIFAR-100, Tiny ImageNet are supported.

## Evaluation examples
```
python3 main.py --gpu 0 --model_structure advGNI --eta 8 --exp_name CIFAR10_test \
    --dataset_name CIFAR10 --resume snapshots/CIFAR10_test/pretrain.pth

```
- resume: Either pretrain.pth (saved at the end of the training) or pretrain_best.pth (early-stopped version) can be evaluated.

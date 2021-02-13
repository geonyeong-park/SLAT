import torch
import torch.nn as nn
import torchvision.transforms as transforms
from utils.utils import mean, std
import os
import sys
import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt
from utils.attack import attack_FGSM, attack_pgd
from utils.utils import clamp, lower_limit, upper_limit
from visualize.visualize_land import compute_perturb, plot_perturb_plt, visualize_perturb
from model.wide_resnet import WideResNet28_10, WideResNet
from autoattack import AutoAttack


def eval(solver, checkpoint, BB_ckpt, eps):
    """
    (1) Visualize loss landscape
    (2) Visualize accumulated penultimate feature
    (2) Test adversarial robustness via FGSM, PGD, Blackbox attack
        - PGD: 50-10
    """
    torch.manual_seed(0)
    np.random.seed(0)

    solver.model.load_state_dict(checkpoint['model'])
    solver.model.eval()
    solver.model.to('cuda')

    BB_model = WideResNet28_10(solver.config).to('cuda')
    BB_model.load_state_dict(BB_ckpt['model'])
    auto_adversary = AutoAttack(AutoWrapper(solver.model), norm='Linf', eps=eps/255., version='standard')
    mu_, std_ = torch.tensor(mean).float(), torch.tensor(std).float()
    denorm = transforms.Normalize((-mu_/std_).tolist(), (1./std_).tolist())

    png_path = lambda x: os.path.join(solver.log_dir, '{}.png'.format(x))

    acc = {
        'FGSM': 0.,
        'PGD': 0.,
        'Black': 0.,
        'auto': 0.
    }
    loss = {
        'FGSM': 0.,
        'PGD': 0.,
        'Black': 0.,
        'auto': 0.
    }
    counter = 0

    for i, (x, y) in enumerate(solver.valid_loader):
        x = x.to('cuda')
        y = y.long().to('cuda')

        # -------------------------
        # (1) Visualize loss landscape
        # -------------------------
        if i == 0:
            adv_vec = attack_FGSM(solver.model, x, y, solver.epsilon, clamp_=False)
            adv_vec = adv_vec[0]
            rademacher_vec = 2.*(torch.randint(2, size=adv_vec.shape)-1.) * solver.epsilon.data.cpu()
            x_ = x[0]
            y_ = y[0]

            rx, ry, zs = compute_perturb(model=solver.model,
                                 image=x_, label=y_,
                                 vec_x=adv_vec, vec_y=rademacher_vec,
                                 range_x=(-1,1), range_y=(-1,1),
                                 grid_size=50,
                                 loss=nn.CrossEntropyLoss(reduction='none'))
            print('computed adversarial loss landscape')
            plot_perturb_plt(rx, ry, zs, png_path, eps,
                             xlabel='Adv', ylabel='Rad',)

        # -------------------------
        # (2) Visualize accumulated perturbation
        # -------------------------
            visualize_perturb(solver.model, x, y, 20, 1.5, 50, png_path)

        # -------------------------
        # (3) Adversarial robustness test
        # -------------------------
        pgd_delta = attack_pgd(solver.model, x, y, solver.epsilon, solver.pgd_alpha, 50, 10)
        FGSM_delta = attack_FGSM(solver.model, x, y, solver.epsilon)
        bb_delta = attack_pgd(BB_model, x, y, solver.epsilon, solver.pgd_alpha, 50, 10)
        auto_sample = auto_adversary.run_standard_evaluation(denorm(x), y, bs=x.shape[0])
        print(auto_sample)

        pgd_loss, pgd_acc = _adv_loss_acc(pgd_delta, x, y, solver.model, solver.cen)
        loss['PGD'] += pgd_loss
        acc['PGD'] += pgd_acc

        FGSM_loss, FGSM_acc = _adv_loss_acc(FGSM_delta, x, y, solver.model, solver.cen)
        loss['FGSM'] += FGSM_loss
        acc['FGSM'] += FGSM_acc

        Black_loss, Black_acc = _adv_loss_acc(bb_delta, x, y, solver.model, solver.cen)
        loss['Black'] += Black_loss
        acc['Black'] += Black_acc

        Auto_loss, Auto_acc = _adv_loss_acc(auto_sample, x, y, AutoWrapper(solver.model), solver.cen, adv_sample=True)
        loss['auto'] += Auto_loss
        acc['auto'] += Auto_acc

        k = y.data.size()[0]
        counter += k

        if i % 1 == 0:
            print('PGD: ', acc['PGD']/counter)
            print('Black: ', acc['Black']/counter)
            print('Auto: ', acc['auto']/counter)

    for k, v in acc.items():
        acc[k] = v / counter
        loss[k] = loss[k] / counter

    print(acc)
    print(loss)

    sample_path = os.path.join(solver.log_dir, 'eval.pkl')
    with open(sample_path, 'wb') as f:
        pkl.dump(acc, f, pkl.HIGHEST_PROTOCOL)
        pkl.dump(loss, f, pkl.HIGHEST_PROTOCOL)

def _adv_loss_acc(delta, x, y, model, cen, adv_sample=False):
    if not adv_sample:
        adv_sample = x + delta[:x.size(0)]
        logit = model(clamp(adv_sample, lower_limit, upper_limit))
    else:
        adv_sample = delta
        logit = model(adv_sample)
    loss = float(cen(logit, y))
    pred = logit.data.max(1)[1]
    acc = float(pred.eq(y.data).cpu().sum())
    return loss, acc

class AutoWrapper(nn.Module):
    def __init__(self, model):
        super(AutoWrapper, self).__init__()
        self.model = model
        self.mu = torch.Tensor([0.4914, 0.4822, 0.4465]).float().view(3, 1, 1).cuda()
        self.sigma = torch.Tensor([0.2471, 0.2435, 0.2616]).float().view(3, 1, 1).cuda()

    def forward(self, x):
        x_ = (x - self.mu) / self.sigma
        return self.model(x_)

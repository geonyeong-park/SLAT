import torch
import torch.nn as nn
import os
import sys
import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt
from utils.attack import attack_FGSM, attack_pgd
from utils.utils import clamp, lower_limit, upper_limit
from visualize.visualize_land import compute_perturb, plot_perturb_plt


def eval(solver, checkpoint, eps):
    """
    (1) Visualize loss landscape
    (2) Test adversarial robustness via FGSM, PGD, Blackbox attack
        - PGD: 50-10
    """
    torch.manual_seed(0)
    np.random.seed(0)

    solver.model.load_state_dict(checkpoint['model'])
    solver.model.eval()
    solver.model.to('cuda')

    png_path = os.path.join(solver.log_dir, 'loss_landscape.png')

    acc = {
        'FGSM': 0.,
        'PGD': 0.,
        'Black': 0.
    }
    loss = {
        'FGSM': 0.,
        'PGD': 0.,
        'Black': 0.
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
                             title='{}_loss_landscape'.format(solver.structure),
                             xlabel='Adv', ylabel='Rad', zlabel='loss')

        # -------------------------
        # (2) Adversarial robustness test
        # -------------------------
        pgd_delta = attack_pgd(solver.model, x, y, solver.epsilon, solver.pgd_alpha, 50, 10)
        FGSM_delta = attack_FGSM(solver.model, x, y, solver.epsilon)

        pgd_logit = solver.model(clamp(x + pgd_delta[:x.size(0)], lower_limit, upper_limit))
        FGSM_logit = solver.model(clamp(x + FGSM_delta[:x.size(0)], lower_limit, upper_limit))

        pgd_loss = solver.cen(pgd_logit, y)
        FGSM_loss = solver.cen(FGSM_loss, y)

        loss['FGSM'] += FGSM_loss.data.cpu().numpy()
        loss['PGD'] += pgd_loss.data.cpu().numpy()

        pgd_pred = pgd_logit.data.max(1)[1]
        FGSM_pred = FGSM_logit.data.max(1)[1]

        acc['FGSM'] += FGSM_pred.eq(y.data).cpu().sum()
        acc['PGD'] += pgd_pred.eq(y.data).cpu().sum()

        k = y.data.size()[0]
        counter += k

    for k, v in acc.items():
        acc[k] = v / counter
        loss[k] = loss[k] / counter

    print(acc)
    print(loss)

import torch
import torch.nn as nn
import os
import sys
import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt
from utils.attack import attack_FGSM, attack_pgd
from utils.utils import clamp, lower_limit, upper_limit


def eval(solver):
    """
    (1) Visualize loss landscape
    (2) Test adversarial robustness via FGSM, PGD, Blackbox attack
        - PGD: 50-10
    """

    # -------------------------
    # (1) Visualize loss landscape
    # -------------------------



    # -------------------------
    # (2) Adversarial robustness test
    # -------------------------
    solver.model.eval()

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

    for x, y in solver.valid_loader:
        x = x.to('cuda')
        y = y.long().to('cuda')

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

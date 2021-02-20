import torch.nn as nn
import torch
import torch.nn.functional as F
import random
from math import sqrt
from model.hidden_module import lower_limit, upper_limit
from utils.utils import clamp

def attack_pgd(model, X, y, epsilon, alpha, attack_iters, restarts, opt=None):
    max_loss = torch.zeros(y.shape[0]).to('cuda')
    max_delta = torch.zeros_like(X).to('cuda')
    for zz in range(restarts):
        delta = torch.zeros_like(X).to('cuda')
        for i in range(len(epsilon)):
            delta[:, i, :, :].uniform_(-epsilon[i][0][0].item(), epsilon[i][0][0].item())
        delta.data = clamp(delta, lower_limit - X, upper_limit - X)
        delta.requires_grad = True
        for _ in range(attack_iters):
            output = model(X + delta)
            index = torch.where(output.max(1)[1] == y)
            if len(index[0]) == 0:
                break
            loss = F.cross_entropy(output, y)
            loss.backward()

            grad = delta.grad.detach()
            d = delta[index[0], :, :, :]
            g = grad[index[0], :, :, :]
            d = clamp(d + alpha * torch.sign(g), -epsilon, epsilon)
            d = clamp(d, lower_limit - X[index[0], :, :, :], upper_limit - X[index[0], :, :, :])
            delta.data[index[0], :, :, :] = d
            delta.grad.zero_()
        all_loss = F.cross_entropy(model(X+delta), y, reduction='none').detach()
        max_delta[all_loss >= max_loss] = delta.detach()[all_loss >= max_loss]
        max_loss = torch.max(max_loss, all_loss)
    return max_delta

def attack_black_simbaODS(model, X, y, attack_iters):
    # modify https://github.com/ermongroup/ODS/blob/master/blackbox_simbaODS.py batchwise
    X_best = torch.autograd.Variable(X.data, requires_grad=True)

    for _ in range(attack_iters):
        output = model(X_best)
        index = torch.where(output.max(1)[1] == y)
        if len(index[0]) == 0:
            break
        loss_best = F.cross_entropy(output, y, reduction='none').detach()

        random_direction = torch.rand_like(output).to('cuda') * 2 - 1
        with torch.enable_grad():
            ODS_loss = (model(X_best[index[0],:,:,:]) * random_direction[index[0]]).sum()
        ODS_loss.backward()
        delta = X_best.grad[index[0],:,:,:].data / X_best.grad[index[0],:,:,:].norm()

        X_new = X_best.clone()
        for sign in [1, -1]:
            X_new[index[0],:,:,:] = X_best[index[0],:,:,:] + 0.2*sign*delta
            X_new = clamp(X_new, lower_limit, upper_limit)
            logits = model(X_new).data
            loss_new = F.cross_entropy(logits, y, reduction='none').detach()
            X_best.data[loss_best<loss_new] = X_new.detach()[loss_best<loss_new]

    return X_best

def attack_FGSM(model, x, y, epsilon, clamp_=True):
    x.requires_grad = True
    logit_clean = model(x)
    loss = nn.CrossEntropyLoss()(logit_clean, y)

    model.zero_grad()
    loss.backward()
    grad = x.grad.data.detach()

    with torch.no_grad():
        sgn_mask = grad.data.sign()

    adv_noise = sgn_mask * epsilon
    if clamp_:
        adv_noise.data = clamp(adv_noise, lower_limit - x, upper_limit - x)
    adv_noise = adv_noise.detach()

    return adv_noise

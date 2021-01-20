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

def attack_FGSM(model, x, y, epsilon):
    x.requires_grad = True
    logit_clean = model(x)
    loss = nn.CrossEntropyLoss()(logit_clean, y)

    model.zero_grad()
    loss.backward()
    grad = x.grad.data.detach()

    with torch.no_grad():
        sgn_mask = grad.data.sign()

    adv_noise = sgn_mask * epsilon
    adv_noise.data = clamp(adv_noise, lower_limit - x, upper_limit - x)
    adv_noise = adv_noise.detach()

    return adv_noise

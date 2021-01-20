import torch.nn as nn
import torch
import torch.nn.functional as F
import random
from math import sqrt

mean = [0.4914, 0.4822, 0.4465]
std = [0.2471, 0.2435, 0.2616]

mu_t = torch.tensor(mean).view(3,1,1).to('cuda')
std_t = torch.tensor(std).view(3,1,1).to('cuda')

upper_limit = ((1. - mu_t)/ std_t)
lower_limit = ((0. - mu_t)/ std_t)

def cos_by_uniform(model, x, y, epsilon, grad=None):
    if grad is None:
        x.requires_grad = True
        logit = model(x)
        loss_ = nn.CrossEntropyLoss()(logit, y)
        loss_.backward()
        grad = x.grad.data

    delta = torch.zeros(x.shape).cuda()
    for j in range(len(epsilon)):
        delta[:, j, :, :].uniform_(-epsilon[j][0][0].item(), epsilon[j][0][0].item())
        delta.data = clamp(delta, lower_limit - x, upper_limit - x)
    delta.requires_grad = True

    delta_output = model(x + delta)
    delta_loss = nn.CrossEntropyLoss()(delta_output, y)

    adv_grad = torch.autograd.grad(delta_loss, delta, create_graph=True)[0]
    cos = cosine_similarity(grad, adv_grad)
    return cos
    #valid_cos += cos.data.cpu().numpy()


def cosine_similarity(grad1, grad2):
    grads_nnz_idx = ((grad1**2).sum([1, 2, 3])**0.5 != 0) * ((grad2**2).sum([1, 2, 3])**0.5 != 0)
    grad1, grad2 = grad1[grads_nnz_idx], grad2[grads_nnz_idx]
    grad1_norms = _l2_norm_batch(grad1)
    grad2_norms = _l2_norm_batch(grad2)
    grad1_normalized = grad1 / grad1_norms[:, None, None, None]
    grad2_normalized = grad2 / grad2_norms[:, None, None, None]
    cos = torch.sum(grad1_normalized * grad2_normalized, (1, 2, 3))
    cos_aggr = cos.mean()
    return cos_aggr

def _l2_norm_batch(v):
    norms = (v ** 2).sum([1, 2, 3]) ** 0.5
    # norms[norms == 0] = np.inf
    return norms

def clamp(X, lower_limit, upper_limit):
    return torch.max(torch.min(X, upper_limit), lower_limit)

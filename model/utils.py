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

class NoisyCNNModule(nn.Module):
    def __init__(self, architecture, eta, alpha_coeff=0., input=False):
        super(NoisyCNNModule, self).__init__()
        self.architecture = architecture
        self.input = input
        self.alpha_coeff = alpha_coeff
        self.eta = eta / std_t if input else torch.tensor(eta).to('cuda')

    def forward(self, x, grad_mask=None, add_adv=False):
        if self.training:
            if self.architecture == 'GNI':
                x_hat = x + torch.randn_like(x) * sqrt(0.001)
                return x_hat
            elif self.architecture == 'advGNI':
                if add_adv:
                    assert grad_mask is not None
                    grad_mask = grad_mask.detach()

                    with torch.no_grad():
                        sgn_mask = grad_mask.data.sign()

                    adv_noise = sgn_mask * self.eta * self.alpha_coeff
                    adv_noise.data = clamp(adv_noise, -self.eta, self.eta)
                    if self.input:
                        adv_noise.data = clamp(adv_noise, lower_limit - x, upper_limit - x)

                    adv_noise = adv_noise.detach()
                    x_hat = x + adv_noise
                    return x_hat
                else:
                    return x
            elif self.architecture == 'FGSM' or self.architecture == 'FGSM_GA':
                if add_adv and self.input:
                    grad_mask = grad_mask.detach()

                    with torch.no_grad():
                        sgn_mask = grad_mask.data.sign()

                    adv_noise = sgn_mask * self.eta
                    adv_noise.data = clamp(adv_noise, lower_limit - x, upper_limit - x)
                    adv_noise = adv_noise.detach()

                    x_hat = x + adv_noise
                    return x_hat
                else:
                    return x
            elif self.architecture == 'FGSM_RS':
                """Wong et al., ICLR 2020"""
                if self.input:
                    if not add_adv:
                        # Initialize delta
                        self.delta = torch.zeros_like(x)
                        for j in range(len(self.eta)):
                            self.delta[:, j, :, :].uniform_(-self.eta[j][0][0].item(), self.eta[j][0][0].item())
                            self.delta.data = clamp(self.delta, lower_limit - x, upper_limit - x)
                        self.delta.requires_grad = True
                        x_hat = x + self.delta[:x.size(0)]
                        return x_hat
                    else:
                        grad = self.delta.grad.detach()
                        self.delta.data = clamp(self.delta + 1.25*self.eta*torch.sign(grad), -self.eta, self.eta)
                        self.delta.data[:x.size(0)] = clamp(self.delta[:x.size(0)], lower_limit - x, upper_limit - x)
                        self.delta = self.delta.detach()
                        x_hat = x + self.delta[:x.size(0)]
                        return x_hat
                else:
                    return x
            else:
                # PGD will be included in noise.py
                return x
        else:
            return x

class PreActBlock(nn.Module):
    '''Pre-activation version of the BasicBlock.'''
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(PreActBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)

        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        if hasattr(self, 'shortcut'):
            shortcut = self.shortcut(x)
        else:
            shortcut = x

        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out += shortcut
        return out


def clamp(X, lower_limit, upper_limit):
    return torch.max(torch.min(X, upper_limit), lower_limit)

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

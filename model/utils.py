import torch.nn as nn
import torch
import torch.nn.functional as F
from math import sqrt


mean = [0.4914, 0.4822, 0.4465]
std = [0.2471, 0.2435, 0.2616]

mu_t = torch.tensor(mean).view(3,1,1).cuda()
std_t = torch.tensor(std).view(3,1,1).cuda()

upper_limit = ((1. - mu_t)/ std_t)
lower_limit = ((0. - mu_t)/ std_t)

class NoisyCNNModule(nn.Module):
    def __init__(self, architecture, eta, input=False):
        super(NoisyCNNModule, self).__init__()
        self.architecture = architecture
        self.input = input
        self.eta = eta / std_t if input else eta

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
                        var_mask = torch.abs(grad_mask.data)

                    adv_noise_raw = torch.abs(torch.randn_like(x)) * self.eta
                    adv_noise = sgn_mask * adv_noise_raw
                    if self.input: adv_noise.data = clamp(adv_noise, lower_limit - x, upper_limit - x)

                    x_hat = x + adv_noise
                    return x_hat
                else:
                    return x
            elif self.architecture == 'FGSM':
                if add_adv and self.input:
                    assert grad_mask is not None
                    grad_mask = grad_mask.detach()

                    with torch.no_grad():
                        sgn_mask = grad_mask.data.sign()

                    adv_noise = sgn_mask * self.eta
                    adv_noise.data = clamp(adv_noise, lower_limit - x, upper_limit - x)

                    x_hat = x + adv_noise
                    return x_hat
                else:
                    return x
            else:
                return x
        else:
            return x


def clamp(X, lower_limit, upper_limit):
    return torch.max(torch.min(X, upper_limit), lower_limit)

def attack_pgd(model, X, y, epsilon, alpha, attack_iters, restarts, opt=None):
    max_loss = torch.zeros(y.shape[0]).cuda()
    max_delta = torch.zeros_like(X).cuda()
    for zz in range(restarts):
        delta = torch.zeros_like(X).cuda()
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

import torch.nn as nn
import torch
import torch.nn.functional as F
from math import sqrt


mean = [0.4914, 0.4822, 0.4465]
std = [0.2471, 0.2435, 0.2616]

mu_t = torch.tensor(mean).view(3,1,1).to('cuda')
std_t = torch.tensor(std).view(3,1,1).to('cuda')

upper_limit = ((1. - mu_t)/ std_t)
lower_limit = ((0. - mu_t)/ std_t)

class NoisyCNNModule(nn.Module):
    def __init__(self, architecture, eta, input=False):
        super(NoisyCNNModule, self).__init__()
        self.architecture = architecture
        self.input = input
        self.eta = eta / std_t if input else torch.tensor(eta).to('cuda')

    def forward(self, x, grad_mask=None, add_adv=False, coeff=None):
        if self.training:
            if self.architecture == 'GNI':
                x_hat = x + torch.randn_like(x) * sqrt(0.001)
                return x_hat
            elif self.architecture == 'advGNI':
                if add_adv:
                    assert grad_mask is not None
                    assert coeff is not None
                    grad_mask = grad_mask.detach()

                    with torch.no_grad():
                        sgn_mask = grad_mask.data.sign()
                        var_mask = torch.abs(grad_mask.data)

                    #adv_noise_raw = torch.abs(0.5*(torch.ones_like(x)+torch.randn_like(x))) * self.eta
                    mu = 0.5*(1-coeff)
                    std = 0.5*(1+coeff)
                    adv_noise_raw = torch.abs(mu*torch.ones_like(x)+std*torch.randn_like(x)) * self.eta

                    adv_noise = sgn_mask * adv_noise_raw
                    adv_noise.data = clamp(adv_noise, -self.eta, self.eta)
                    if self.input:
                        adv_noise.data = clamp(adv_noise, lower_limit - x, upper_limit - x)
                    adv_noise = adv_noise.detach()

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

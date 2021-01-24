import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.distributions import MultivariateNormal
from torch.autograd import Variable
from math import sin, cos, sqrt


class basemodel(nn.Module):
    def __init__(self, n_hidden, architecture, epsilon):
        super(basemodel, self).__init__()
        self.architecture = architecture
        self.epsilon = epsilon

        self.sigmoid = nn.Sigmoid()
        self.fc = nn.Linear(2, n_hidden)
        self.logit = nn.Linear(n_hidden, 1, bias=False)

        self.noisy_module = nn.ModuleDict({
            'input': HiddenPerturb(self.architecture, self.epsilon, 1., True),
            'fc': HiddenPerturb(self.architecture, self.epsilon, 3.5),
        })
        # epsilon=0.1, alpha_1=1., alpha_2=3.5
        # FGSM=96.69%, advGNI=94.19%
        # python3 gaussian_test.py --gpu 2 --n_hidden 512 --architecture advGNI --epsilon 0.1

        self.grads = {
            'input': None,
            'fc': None,
        }

    def save_grad(self, name):
        def hook(grad):
            self.grads[name] = grad
        return hook

    def forward(self, x, add_adv=False, hook=False):
        x_hat = self.noisy_module['input'](x, self.grads['input'], add_adv)

        h = self.fc(x_hat)
        h = self.sigmoid(h)

        if hook:
            h.register_hook(self.save_grad('fc'))
        h = self.noisy_module['fc'](h, self.grads['fc'], add_adv)

        out = self.logit(h).view(h.shape[0])
        return out


class HiddenPerturb(nn.Module):
    def __init__(self, architecture, eta, alpha_coeff=0.5, input=False):
        super(HiddenPerturb, self).__init__()
        self.architecture = architecture
        self.input = input
        self.alpha_coeff = alpha_coeff
        self.eta = torch.tensor(eta).to('cuda')

    def forward(self, x, grad_mask=None, add_adv=False):
        if self.training:
            if self.architecture == 'GNI':
                x_hat = x + torch.randn_like(x) * sqrt(0.1)
                return x_hat
            elif self.architecture == 'advGNI':
                if add_adv:
                    assert grad_mask is not None
                    grad_mask = grad_mask.detach()

                    with torch.no_grad():
                        sgn_mask = grad_mask.data.sign()

                    adv_noise = sgn_mask * self.eta * self.alpha_coeff
                    #if self.input:
                    #    adv_noise.data = clamp(adv_noise, lower_limit - x, upper_limit - x)

                    adv_noise = adv_noise.detach()
                    x_hat = x + adv_noise
                    return x_hat
                else:
                    return x

            elif self.architecture == 'FGSM':
                if add_adv and self.input:
                    grad_mask = grad_mask.detach()

                    with torch.no_grad():
                        sgn_mask = grad_mask.data.sign()

                    adv_noise = sgn_mask * self.eta
                    #adv_noise.data = clamp(adv_noise, lower_limit - x, upper_limit - x)
                    adv_noise = adv_noise.detach()

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

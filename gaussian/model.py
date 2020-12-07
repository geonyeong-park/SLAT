import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.distributions import MultivariateNormal
from torch.autograd import Variable
from math import sin, cos, sqrt


class basemodel(nn.Module):
    def __init__(self, n_hidden, covariance):
        super(basemodel, self).__init__()

        self.sigmoid = nn.Sigmoid()
        self.fc = nn.Linear(2, n_hidden)
        self.logit = nn.Linear(n_hidden, 2)

        self.noise_L1 = nn.Linear(2, 2, bias=False)
        self.noise_L1 = self.noise_L1.apply(self.tril_init) if covariance else self.noise_L1.apply(self.eye_init)
        self.noise_L2 = nn.Linear(n_hidden, n_hidden, bias=False)
        self.noise_L2 = self.noise_L2.apply(self.tril_init) if covariance else self.noise_L2.apply(self.eye_init)

        self.mask = lambda x: torch.tril(torch.ones_like(x)).to('cuda') if covariance else torch.eye(x.size(0)).to('cuda')
        self.noise_L1.weight.register_hook(self.get_zero_grad_hook(self.mask(self.noise_L1.weight)))
        self.noise_L2.weight.register_hook(self.get_zero_grad_hook(self.mask(self.noise_L2.weight)))

    def tril_init(self, m):
        if isinstance(m, nn.Linear):
            with torch.no_grad():
                m.weight.copy_(torch.tril(m.weight))

    def eye_init(self, m):
        if isinstance(m, nn.Linear):
            with torch.no_grad():
                m.weight.copy_(torch.eye(m.weight.size(0)))

    # Zero out gradients
    def get_zero_grad_hook(self, mask):
        def hook(grad):
            return grad * mask
        return hook

    def forward(self, x, use_adv=True):
        if self.training:
            if use_adv:
                norm_penalty = torch.tensor(0., device='cuda')
                x_noise = self.noise_L1(torch.randn_like(x)).to('cuda')
                x_hat = x+x_noise
                norm_penalty += torch.norm(x_noise).to('cuda')

                h = self.fc(x_hat)
                h = self.sigmoid(h)
                h_noise = self.noise_L2(torch.randn_like(h)).to('cuda')
                h_hat = h + h_noise
                norm_penalty += torch.norm(h_noise).to('cuda')

                logit = self.logit(h_hat)

            else:
                x_noise = sqrt(0.1)*torch.randn_like(x).to('cuda')
                x_hat = x+x_noise

                h1 = self.fc(x_hat)
                h1 = self.sigmoid(h1)
                h1_noise = sqrt(0.1)*torch.randn_like(h1).to('cuda')
                h1 = h1 + h1_noise

                logit = self.logit(h1)
                return logit
            return logit, norm_penalty

        else:
            h1 = self.fc(x)
            h2 = self.sigmoid(h1)
            logit = self.logit(h2)
            return logit, h2

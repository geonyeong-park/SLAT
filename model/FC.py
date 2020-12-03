import torch.nn as nn
import torch.nn.functional as F
import math
from math import sqrt
import torch.utils.model_zoo as model_zoo
import torch
import torchvision
from torchvision import models
from torchvision.models.resnet import ResNet, Bottleneck, model_urls, load_state_dict_from_url
import numpy as np


class NoisyFC(nn.Module):
    def __init__(self, config):
        super(NoisyFC, self).__init__()
        self.config = config
        self.data_name = config['dataset']['name']
        self.num_cls = config['dataset'][self.data_name]['num_cls']
        self.input_size = config['dataset'][self.data_name]['input_size']
        self.channel = config['dataset'][self.data_name]['channel']
        self.architecture = config['model']['baseline']
        self.use_adversarial_noise = True if 'Adv' in self.architecture else False

        self.module_output = self.config['model']['FC']['module']
        self.module_input = [self.input_size**2*self.channel] + self.module_output[:-1]
        noise_linear = self.config['model']['FC']['noise_linear']
        noise_num_layer = self.config['model']['FC']['noise_num_layer']
        norm = self.config['model']['FC']['norm']

        self.noisy_module = nn.ModuleList([
            NoisyModule(self.architecture, noise_linear, noise_num_layer, i, j, norm)
         for i,j in zip(self.module_input, self.module_output)])

        self.logit = nn.Linear(self.module_output[-1], self.num_cls)

    def forward(self, x):
        h = x
        self.norm_penalty = torch.tensor(0.).to('cuda')
        for i in range(len(self.noisy_module)):
            h = self.noisy_module[i](h)
            self.norm_penalty += self.noisy_module[i].norm_penalty

        logit = self.logit(h)
        return logit

class NoisyModule(nn.Module):
    def __init__(self, architecture, noise_linear, noise_num_layer, in_unit, out_unit, norm):
        super(NoisyModule, self).__init__()
        self.architecture = architecture
        self.noise_linear = noise_linear
        self.noise_num_layer = noise_num_layer
        self.in_unit = in_unit
        self.out_unit = out_unit
        self.norm = norm

        self.w = nn.Linear(in_unit, out_unit)
        self.ReLU = nn.ReLU()

        if self.noise_linear:
            assert noise_num_layer == 1
            self._linear_noise()
        else:
            self._nonlinear_noise()

    def _linear_noise(self):
        self.noise_layer = nn.Linear(self.in_unit, self.in_unit, bias=not self.noise_linear)
        self.noise_layer.apply(self._tril_init)
        self.mask = torch.tril(torch.ones((self.in_unit, self.in_unit))).to('cuda')
        self.noise_layer.weight.register_hook(self._get_zero_grad_hook(self.mask))

    def _nonlinear_noise(self):
        module = []
        for _ in range(self.noise_num_layer-2):
            module.append(nn.Linear(self.in_unit, self.in_unit, bias=True))
            module.append(self.ReLU)
        module.append(nn.Linear(self.in_unit, self.in_unit, bias=True))
        self.noise_layer = nn.Sequential(*module)

    def _tril_init(self, m):
        if isinstance(m, nn.Linear):
            with torch.no_grad():
                m.weight.copy_(torch.tril(m.weight))

    # Zero out gradients
    def _get_zero_grad_hook(self, mask):
        def hook(grad):
            return grad * mask
        return hook

    def forward(self, x):
        self.norm_penalty = torch.tensor(0.).to('cuda')
        if self.training:
            if self.architecture == 'GNI':
                x = x + torch.randn_like(x) * sqrt(0.1)
                h = self.w(x)
                h = self.ReLU(h)
                return h
            elif self.architecture == 'advGNI':
                x_noise = self.noise_layer(sqrt(0.1)*torch.randn_like(x))
                x_hat = x+x_noise
                self.norm_penalty += torch.mean(torch.norm(x_noise, float(self.norm), dim=1)).to('cuda')

                h = self.w(x_hat)
                h = self.ReLU(h)
                return h
            else:
                h = self.w(x)
                h = self.ReLU(h)
                return h
        else:
            h = self.w(x)
            h = self.ReLU(h)
            return h

"""
class NormalFC(NoisyFC):
    def __init__(self, config):
        super(NormalFC, self).__init__(config)
        self.normal_module = nn.Sequential(*[
            NormalModule(i, j, config['model']['GNI_in_normal'])
        for i,j in zip(self.module_input, self.module_output)])
        self.use_adversarial_noise = False

    def forward(self, x):
        h = self.normal_module(x)
        logit = self.logit(h)
        return logit

class NormalModule(nn.Module):
    def __init__(self, in_unit, out_unit, add_noise=False):
        super(NormalModule, self).__init__()
        self.in_unit = in_unit
        self.out_unit = out_unit
        self.add_noise = add_noise

        self.linear_layer = nn.Linear(in_unit, out_unit)
        self.ReLU = nn.ReLU()

    def forward(self, x):
        if self.add_noise and self.training:
            x = x + torch.randn_like(x) * sqrt(0.1)
        h = self.linear_layer(x)
        h = self.ReLU(h)
        return h
"""


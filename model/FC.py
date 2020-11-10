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

        self.module_output = self.config['model']['FC']['module']
        self.module_input = [self.input_size**2*self.channel] + self.module_output[:-1]
        self.noise_L = self.config['model']['FC']['noise_L']

        self.noisy_module = nn.ModuleList([
            NoisyModule(self.noise_L, i, j)
         for i,j in zip(self.module_input, self.module_output)])

        self.logit = nn.Linear(self.module_output[-1], self.num_cls)

    def forward(self, x):
        h = x
        norm_penalty = torch.tensor(0.).to('cuda')
        for i in range(len(self.noisy_module)):
            h = self.noisy_module[i](h)
            norm_penalty += self.noisy_module[i].norm_penalty

        logit = self.logit(h)
        return logit, norm_penalty

class NoisyModule(nn.Module):
    def __init__(self, noise_L, in_unit, out_unit):
        super(NoisyModule, self).__init__()
        self.noise_L = noise_L
        self.in_unit = in_unit
        self.out_unit = out_unit

        self.linear_layer = nn.Linear(in_unit, out_unit)
        self.noise_layer = nn.Linear(in_unit, in_unit, bias=not noise_L)
        self.ReLU = nn.ReLU()

        if self.noise_L:
            self.noise_layer.apply(self._tril_init)
            self.mask = torch.tril(torch.ones((in_unit, in_unit))).to('cuda')
            self.noise_layer.weight.register_hook(self._get_zero_grad_hook(self.mask))

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
            x_noise = self.noise_layer(torch.randn_like(x))
            x_hat = x+x_noise
            self.norm_penalty += torch.norm(x_noise).to('cuda')

            h = self.linear_layer(x_hat)
            h = self.ReLU(h)
            return h
        else:
            h = self.linear_layer(x)
            h = self.ReLU(h)
            return h

class NormalFC(NoisyFC):
    def __init__(self, config):
        super(NormalFC, self).__init__(config)
        self.normal_module = nn.Sequential(*[
            NormalModule(i, j, config['model']['add_noise'])
        for i,j in zip(self.module_input, self.module_output)])

    def forward(self, x):
        h = self.normal_module(x)
        logit = self.logit(h)
        return logit, None

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


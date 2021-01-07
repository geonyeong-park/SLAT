import torch.nn as nn
import torch
from torchvision import models
from model.utils import NoisyCNNModule, PreActBlock, mean, std
import torch.nn.functional as F

# PreActResNet Code is largely based on:
# https://github.com/locuslab/fast_adversarial/tree/master/CIFAR10

class PreActResNet(nn.Module):
    def __init__(self, block, num_blocks, config):
        super(PreActResNet, self).__init__()

        self.config = config
        self.data_name = config['dataset']['name']
        self.num_cls = config['dataset'][self.data_name]['num_cls']
        self.input_size = config['dataset'][self.data_name]['input_size']
        self.architecture = config['model']['baseline']
        self.eta = self.config['model']['ResNet']['eta']

        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.bn = nn.BatchNorm2d(512 * block.expansion)
        self.linear = nn.Linear(512 * block.expansion, self.num_cls)

        self.noisy_module = nn.ModuleDict({
            'input': NoisyCNNModule(self.architecture, self.eta/255., True, 3, size=self.input_size),
            'conv1': NoisyCNNModule(self.architecture, self.eta/255., indim=64, size=self.input_size),
            'layer1': NoisyCNNModule(self.architecture, self.eta/255., indim=64, size=self.input_size),
            'layer2': NoisyCNNModule(self.architecture, self.eta/255., indim=128, size=self.input_size//2),
            'layer3': NoisyCNNModule(self.architecture, self.eta/255., indim=256, size=self.input_size//4),
            'layer4': NoisyCNNModule(self.architecture, self.eta/255., indim=512, size=self.input_size//8),
        })

        self.grads = {
            'input': None,
            'conv1': None,
            'layer1': None,
            'layer2': None,
            'layer3': None,
            'layer4': None,
        }

    def save_grad(self, name):
        def hook(grad):
            self.grads[name] = grad
        return hook

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, add_adv=False, hook=False, detach_noise=False):
        x_hat = self.noisy_module['input'](x, self.grads['input'], add_adv, detach_noise)

        h = self.conv1(x_hat)
        if hook:
            h.register_hook(self.save_grad('conv1'))
        h = self.noisy_module['conv1'](h, self.grads['conv1'], add_adv, detach_noise)

        h = self.layer1(h)
        if hook:
            h.register_hook(self.save_grad('layer1'))
        h = self.noisy_module['layer1'](h, self.grads['layer1'], add_adv, detach_noise)

        h = self.layer2(h)
        if hook:
            h.register_hook(self.save_grad('layer2'))
        h = self.noisy_module['layer2'](h, self.grads['layer2'], add_adv, detach_noise)

        h = self.layer3(h)
        if hook:
            h.register_hook(self.save_grad('layer3'))
        h = self.noisy_module['layer3'](h, self.grads['layer3'], add_adv, detach_noise)

        h = self.layer4(h)
        if hook:
            h.register_hook(self.save_grad('layer4'))
        h = self.noisy_module['layer4'](h, self.grads['layer4'], add_adv, detach_noise)

        h = F.relu(self.bn(h))
        h = F.avg_pool2d(h, 4)
        h = h.view(h.size(0), -1)
        h = self.linear(h)
        return h

    def _yield_theta(self):
        b = []
        b.append(self.conv1)
        b.append(self.layer1)
        b.append(self.layer2)
        b.append(self.layer3)
        b.append(self.layer4)
        b.append(self.bn)
        b.append(self.linear)

        for i in range(len(b)):
            for j in b[i].modules():
                for k in j.parameters():
                    if k.requires_grad:
                        yield k

    def _yield_phi(self):
        b = []
        b.append(self.noisy_module)

        for i in range(len(b)):
            for j in b[i].modules():
                jj = 0
                for k in j.parameters():
                    jj += 1
                    if k.requires_grad:
                        yield k

    def optim_theta(self):
        return [{'params': self._yield_theta()}]

    def optim_phi(self):
        return [{'params': self._yield_phi()}]


def PreActResNet18(config):
    model = PreActResNet(PreActBlock, [2,2,2,2], config)
    return model


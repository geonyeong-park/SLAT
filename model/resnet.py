import torch.nn as nn
import torch
from torchvision import models
from model.utils import NoisyCNNModule, mean, std
import torch.nn.functional as F

# PreActResNet Code is largely based on:
# https://github.com/locuslab/fast_adversarial/tree/master/CIFAR10

class PreActResNet(nn.Module):
    def __init__(self, block, num_blocks, config):
        super(PreActResNet, self).__init__()

        self.config = config
        self.data_name = config['dataset']['name']
        self.num_cls = config['dataset'][self.data_name]['num_cls']
        self.architecture = config['model']['baseline']
        eta = self.config['model']['ResNet']['eta']

        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.bn = nn.BatchNorm2d(512 * block.expansion)
        self.linear = nn.Linear(512 * block.expansion, self.num_cls)

        self.noisy_module = nn.ModuleList([
            NoisyCNNModule(self.architecture, eta/255., input=True),
            NoisyCNNModule(self.architecture, 0.75*eta/255.),
            NoisyCNNModule(self.architecture, 0.75*eta/255.),
            NoisyCNNModule(self.architecture, 0.75*eta/255.),
            NoisyCNNModule(self.architecture, 0.75*eta/255.),
        ])
        self.noisy_module_name = ['conv1', 'layer1', 'layer2', 'layer3']

        self.grads = {}

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

    def forward(self, x, grad_mask=None, add_adv=False, hook=False, coeff=None):
        if not add_adv: grad_mask = [None]*len(self.noisy_module)
        else:
            if grad_mask is not None: grad_mask = grad_mask + [None]*(len(self.noisy_module)-len(grad_mask))
            else: grad_mask = [None]*len(self.noisy_module)
        x_hat = self.noisy_module[0](x, grad_mask[0], add_adv, coeff)

        h = self.conv1(x_hat)
        if self.architecture == 'advGNI' and hook:
            h.retain_grad()
            h.register_hook(self.save_grad(self.noisy_module_name[0]))
        h = self.noisy_module[1](h, grad_mask[1], add_adv, coeff)

        h = self.layer1(h)
        if self.architecture == 'advGNI' and hook:
            h.retain_grad()
            h.register_hook(self.save_grad(self.noisy_module_name[1]))
        h = self.noisy_module[2](h, grad_mask[2], add_adv, coeff)

        h = self.layer2(h)
        if self.architecture == 'advGNI' and hook:
            h.retain_grad()
            h.register_hook(self.save_grad(self.noisy_module_name[2]))
        h = self.noisy_module[3](h, grad_mask[3], add_adv, coeff)

        h = self.layer3(h)
        if self.architecture == 'advGNI' and hook:
            h.retain_grad()
            h.register_hook(self.save_grad(self.noisy_module_name[3]))
        h = self.noisy_module[4](h, grad_mask[4], add_adv, coeff)

        h = self.layer4(h)

        h = F.relu(self.bn(h))
        h = F.avg_pool2d(h, 4)
        h = h.view(h.size(0), -1)
        h = self.linear(h)
        return h


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
        shortcut = self.shortcut(x) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out += shortcut
        return out


class PreActBottleneck(nn.Module):
    '''Pre-activation version of the original Bottleneck module.'''
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(PreActBottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)

        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out = self.conv3(F.relu(self.bn3(out)))
        out += shortcut
        return out


def PreActResNet18(config):
    return PreActResNet(PreActBlock, [2,2,2,2], config)


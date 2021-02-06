import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.hidden_module import HiddenPerturb

# Modified from https://github.com/bearpaw/pytorch-classification/blob/master/models/cifar/wrn.py
class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, dropRate=0.0):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.droprate = dropRate
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                               padding=0, bias=False) or None
    def forward(self, x):
        if not self.equalInOut:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        out = self.relu2(self.bn2(self.conv1(out if self.equalInOut else x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.conv2(out)
        return torch.add(x if self.equalInOut else self.convShortcut(x), out)

class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, dropRate=0.0):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride, dropRate)
    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, dropRate):
        layers = []
        for i in range(nb_layers):
            layers.append(block(i == 0 and in_planes or out_planes, out_planes, i == 0 and stride or 1, dropRate))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)

class WideResNet(nn.Module):
    def __init__(self, config, depth=28, widen_factor=10, dropRate=0.0, fc_input_dim_scale=1, eta=None):
        super(WideResNet, self).__init__()

        self.config = config
        self.data_name = config['dataset']['name']
        num_classes = config['dataset'][self.data_name]['num_cls']
        self.architecture = config['model']['baseline']

        if eta is None:
            self.eta = self.config['model']['ResNet']['eta']
        else:
            self.eta = eta

        nChannels = [16, 16*widen_factor, 32*widen_factor, 64*widen_factor]
        assert (depth - 4) % 6 == 0, 'depth should be 6n+4'
        n = (depth - 4) // 6
        block = BasicBlock

        # 1st conv before any network block
        self.conv1 = nn.Conv2d(3, nChannels[0], kernel_size=3, stride=1,
                               padding=1, bias=False)
        # 1st block
        self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1, dropRate)

        # 2nd block
        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 2, dropRate)

        # 3rd block
        self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 2, dropRate)

        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(nChannels[3])
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(fc_input_dim_scale*nChannels[3], num_classes)
        self.nChannels = fc_input_dim_scale*nChannels[3]

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

        if self.architecture == 'advGNI':
            self.coeff_lower = 0.5
            self.coeff_higher = 1.
        elif self.architecture == 'advGNI_GA':
            self.coeff_lower, self.coeff_higher = 0.6, 0.6
        else:
            self.coeff_lower, self.coeff_higher = 0., 0.

        self.noisy_module = nn.ModuleDict({
            'input': HiddenPerturb(self.architecture, self.eta/255., self.coeff_lower, True),
            'conv1': HiddenPerturb(self.architecture, self.eta/255., self.coeff_lower),
            'block1': HiddenPerturb(self.architecture, self.eta/255, self.coeff_lower),
            'block2': HiddenPerturb(self.architecture, self.eta/255., self.coeff_lower),
            'block3': HiddenPerturb(self.architecture, self.eta/255., self.coeff_higher),
        })

        self.grads = {
            'input': None,
            'conv1': None,
            'block1': None,
            'block2': None,
            'block3': None,
        }

    def save_grad(self, name):
        def hook(grad):
            self.grads[name] = grad
        return hook

    def forward(self, x, add_adv=False, hook=False, return_hidden=False):
        x_hat = self.noisy_module['input'](x, self.grads['input'], add_adv)

        h = self.conv1(x_hat)
        if hook:
            h.register_hook(self.save_grad('conv1'))
        h = self.noisy_module['conv1'](h, self.grads['conv1'], add_adv)

        h = self.block1(h)
        if hook:
            h.register_hook(self.save_grad('block1'))
        h = self.noisy_module['block1'](h, self.grads['block1'], add_adv)

        h = self.block2(h)
        if hook:
            h.register_hook(self.save_grad('block2'))
        h = self.noisy_module['block2'](h, self.grads['block2'], add_adv)

        h = self.block3(h)
        if hook:
            h.register_hook(self.save_grad('block3'))
        h = self.noisy_module['block3'](h, self.grads['block3'], add_adv)
        hidden = h
        #print('min: {}, max: {}, mean: {}'.format(h.view(h.shape[0], -1).min(1).values, h.view(h.shape[0], -1).max(1).values, h.view(h.shape[0], -1).mean(dim=1).values))

        out = self.relu(self.bn1(h))
        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.nChannels)
        if not return_hidden:
            return self.fc(out)
        else:
            return self.fc(out), hidden

def WideResNet28_10(config):
    model = WideResNet(config, depth=28, widen_factor=10, dropRate=0.3)
    return model

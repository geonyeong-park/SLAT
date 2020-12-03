import torch.nn as nn
import torch
import torch.nn.functional as F
from model.FC import NoisyModule
from math import sqrt


class NoisyAllConv(nn.Module):
    def __init__(self, config):
        super(NoisyAllConv, self).__init__()

        self.config = config
        self.data_name = config['dataset']['name']
        self.num_cls = config['dataset'][self.data_name]['num_cls']
        self.input_size = config['dataset'][self.data_name]['input_size']
        self.channel = config['dataset'][self.data_name]['channel']
        self.architecture = config['model']['baseline']
        self.use_adversarial_noise = True if 'Adv' in self.architecture else False
        noise_linear = self.config['model']['FC']['noise_linear']
        noise_num_layer = self.config['model']['FC']['noise_num_layer']
        norm = self.config['model']['FC']['norm']
        self.ReLU = nn.ReLU()

        self.m1 = nn.Sequential(*[
            nn.Conv2d(self.channel, 96, 3, padding=1),
            self.ReLU,
            nn.Conv2d(96, 96, 3, padding=1),
            self.ReLU,
            nn.Conv2d(96, 96, 3, padding=1, stride=2),
            self.ReLU,
        ])

        self.m2 = nn.Sequential(*[
            nn.Conv2d(96, 192, 3, padding=1),
            self.ReLU,
            nn.Conv2d(192, 192, 3, padding=1),
            self.ReLU,
            nn.Conv2d(192, 192, 3, padding=1, stride=2),
            self.ReLU,
        ])

        self.m3 = nn.Sequential(*[
            nn.Conv2d(192, 192, 3, padding=1),
            self.ReLU,
            nn.Conv2d(192, 192, 1),
            self.ReLU
        ])

        self.class_conv = nn.Conv2d(192, self.num_cls, 1)

        self.noisy_module = nn.ModuleList(*[
            NoisyCNNModule(self.architecture, noise_linear, noise_num_layer, 3, norm,
                           image=True, channel=self.channel),
            NoisyCNNModule(self.architecture, noise_linear, noise_num_layer, 96, norm),
            NoisyCNNModule(self.architecture, noise_linear, noise_num_layer, 192, norm),
        ])


    def forward(self, x):
        x_hat = self.noisy_module[0](x)
        h = self.m1(x_hat)
        h_ = self.noisy_module[1](h)

        h = self.m2(h_)
        h_ = self.noisy_module[2](h)

        h = self.m3(h_)
        h = self.class_conv(h)

        pool_out = F.adaptive_avg_pool2d(h, 1)
        pool_out.view(-1, self.num_cls)
        return pool_out

class NoisyCNNModule(NoisyModule):
    def __init__(self, architecture, noise_linear, noise_num_layer, in_unit, norm,
                 image=False, channel=3):
        super(NoisyCNNModule, self).__init__(architecture, noise_linear, noise_num_layer,
                                             in_unit, in_unit, norm)
        del self.w
        self.image = image
        self.channel = channel

        if self.image:
            self._image_noise()

        if self.architecture == 'dropout': self.drop = nn.Dropout()

    def _image_noise(self):
        self.noise_layer = nn.Conv2d(self.channel, self.channel, 1)

    def forward(self, x):
        self.norm_penalty = torch.tensor(0.).to('cuda')
        if self.training:
            if self.architecture == 'GNI':
                x_hat = x + torch.randn_like(x) * sqrt(0.1)
                return x_hat
            elif self.architecture == 'advGNI':
                noise_shape = x.shape if self.image else x.shape[ :2]
                noise = self.noise_layer(sqrt(0.1)*torch.randn(noise_shape))
                if not self.image: noise = noise.view(-1,self.in_unit,1,1).repeat(1,1,x.shape[-2],x.shape[-1])
                x_hat = x + noise
                self.norm_penalty += torch.mean(torch.norm(noise, float(self.norm), dim=1)).to('cuda')
                return x_hat
            elif self.architecture == 'dropout':
                x_hat = self.drop(x) if not self.image else x
                return x_hat
            else:
                return x
        else:
            if self.architecture == 'dropout': return self.drop(x) if not self.image else x
            return x




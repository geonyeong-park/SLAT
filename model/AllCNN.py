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
        self.use_adversarial_noise = True if 'advGNI' in self.architecture else False
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

        self.noisy_module = nn.ModuleList([
            NoisyCNNModule(self.architecture, noise_linear, noise_num_layer, 3, self.input_size, norm, True, self.channel),
            NoisyCNNModule(self.architecture, noise_linear, noise_num_layer, 96, self.input_size//2, norm),
            NoisyCNNModule(self.architecture, noise_linear, noise_num_layer, 192, self.input_size//4, norm),
        ])

    def forward(self, x):
        self.norm_penalty = torch.tensor(0.).to('cuda')
        x_hat = self.noisy_module[0](x)
        self.norm_penalty += self.noisy_module[0].norm_penalty

        h = self.m1(x_hat)
        h_ = self.noisy_module[1](h)
        self.norm_penalty += self.noisy_module[1].norm_penalty

        h = self.m2(h_)
        h_ = self.noisy_module[2](h)
        self.norm_penalty += self.noisy_module[2].norm_penalty

        h = self.m3(h_)
        h = self.class_conv(h)

        pool_out = F.adaptive_avg_pool2d(h, 1)
        pool_out = pool_out.view(-1, self.num_cls)
        return pool_out

class NoisyCNNModule(NoisyModule):
    def __init__(self, architecture, noise_linear, noise_num_layer, in_unit, in_wh, norm,
                 image=False, channel=3):
        super(NoisyCNNModule, self).__init__(architecture, noise_linear, noise_num_layer,
                                             in_unit, in_unit, norm)
        del self.w
        self.image = image
        self.channel = channel
        self.in_wh = in_wh

        if self.architecture == 'dropout': self.drop = nn.Dropout()

        if self.architecture == 'advGNI':
            if self.noise_linear:
                assert noise_num_layer == 1
                self.spatial_noise_layer = self._linear_noise(in_wh*in_wh)
                self.spatial_noise_layer.apply(self._tril_init)
                self.mask = torch.tril(torch.ones((in_wh**2, in_wh**2))).to('cuda')
                self.spatial_noise_layer.weight.register_hook(self._get_zero_grad_hook(self.mask))
            else:
                self.spatial_noise_layer = self._nonlinear_noise(in_wh*in_wh)

    def forward(self, x):
        self.norm_penalty = torch.tensor(0.).to('cuda')
        if self.training:
            if self.architecture == 'GNI':
                x_hat = x + torch.randn_like(x) * sqrt(0.01)
                return x_hat
            elif self.architecture == 'advGNI':
                noise_shape_ch, noise_shape_wh = x.shape[ :2], (x.shape[0], self.in_wh*self.in_wh)
                noise_wh = self.spatial_noise_layer(sqrt(0.01)*torch.randn(noise_shape_wh).to('cuda'))
                noise_wh = noise_wh.view(-1, 1, self.in_wh, self.in_wh).repeat(1,x.shape[1],1,1)

                if not self.image:
                    noise_ch = self.noise_layer(sqrt(0.01)*torch.randn(noise_shape_ch).to('cuda'))
                    noise_ch = noise_ch.view(-1,self.in_unit,1,1).repeat(1,1,x.shape[-2],x.shape[-1])
                    self.norm_penalty += torch.mean(torch.norm(noise_ch, float(self.norm), dim=1)).to('cuda')
                    x_hat = x + (noise_ch + noise_wh) / 2.
                else:
                    x_hat = x + noise_wh
                self.norm_penalty += torch.mean(torch.norm(noise_wh, float(self.norm), dim=1)).to('cuda')
                return x_hat

            elif self.architecture == 'dropout':
                x_hat = self.drop(x) if not self.image else x
                return x_hat
            else:
                return x
        else:
            if self.architecture == 'dropout': return self.drop(x) if not self.image else x
            return x




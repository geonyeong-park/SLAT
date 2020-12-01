import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.nn as nn
from torch.distributions import MultivariateNormal
from torch.autograd import Variable
import torchvision
import os
import numpy as np
import matplotlib.pyplot as plt
from math import sin, cos, sqrt
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
import argparse

from gaussian.Data import DataGen
from gaussian.model import basemodel
from gaussian.Visualize_ import draw_decision_countour, draw_eigen

np.random.seed(3)
torch.manual_seed(5)


parser = argparse.ArgumentParser(description="Generalization by Noise")
parser.add_argument("--ld", type=float, default=1e-3,
                    required=False, help="Lagrangian Multiplier for L2 penalty")
parser.add_argument("--n_hidden", type=int, nargs='+', default=None, required=True,
                    help="number of hidden neurons.")
parser.add_argument("--eval", default=False, action='store_true', required=False,
                    help="eval mode.")
parser.add_argument("--gpu", type=int, nargs='+', default=None, required=True,
                    help="choose gpu device.")

args = parser.parse_args()


gpus_tobe_used = ','.join([str(gpuNum) for gpuNum in args.gpu])
print('gpus_tobe_used: {}'.format(gpus_tobe_used))
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpus_tobe_used)

cudnn.enabled = True
cudnn.benchmark = True


n_samples = 1000 # Per domain
data_noise = 0.2
data_format = 'uniform' # uniform, twomoon, circles
neural_noise_init = 0.1

lr = 1e-3
use_adv = True
epoch = 1000

ld = args.ld
n_hidden = args.n_hidden[0]
if args.eval:
    n_hidden = args.n_hidden

# ---------------------------------------------------------
data_loader = DataGen(n_samples, data_noise, data_format)

X = torch.tensor(data_loader.X_train).float().to('cuda')
Y = torch.tensor(data_loader.y_train).long().to('cuda')
X_test = torch.tensor(data_loader.X_test).float().to('cuda')
Y_test = torch.tensor(data_loader.y_test).long().to('cuda')
cen=torch.nn.CrossEntropyLoss()

def train():
    encoder = basemodel(n_hidden).to('cuda')

    opt_theta = torch.optim.SGD(list(encoder.fc.parameters())+list(encoder.logit.parameters()), lr=lr, momentum=0.9)
    opt_noise  = torch.optim.SGD(list(encoder.noise_L1.parameters())+list(encoder.noise_L2.parameters()), lr=lr, momentum=0.9)
    encoder.train()

    for e in range(epoch):
        if use_adv:
            logit, _ = encoder(X)

            opt_theta.zero_grad()
            theta_loss = cen(logit, Y)
            theta_loss.backward()
            opt_theta.step()

            opt_noise.zero_grad()
            logit, norm = encoder(X)
            noise_loss = -cen(logit, Y) + ld*norm
            noise_loss.backward()
            opt_noise.step()
        else:
            logit = encoder(X, use_adv=False)
            opt_theta.zero_grad()
            theta_loss = cen(logit, Y)
            theta_loss.backward()
            opt_theta.step()

    encoder.eval()
    acc = validation(X_test, Y_test, encoder)
    print('[Lambda: {}]/[n_hidden: {}] || Acc: {}'.format(ld, n_hidden, acc))

    torch.save({
        'model': encoder.state_dict(),
    }, os.path.join('snapshots', 'gaussian_test', '{}_{}_{}.pth'.format(n_hidden, ld, use_adv)))
    print('taking snapshot ...')


def validation(x_test, y_test, encoder):
    logit, _ = encoder(x_test)
    pred = logit.data.max(1).indices
    correct = pred.eq(y_test.data).cpu().sum()
    acc = 100.*correct/len(y_test)
    return acc

def eval(n_hidden_list):
    ckpt = lambda x,y: os.path.join('snapshots', 'gaussian_test', '{}_{}.pth'.format(x, y))

    fig_bdry, ax_bdry = plt.subplots(1, len(n_hidden_list), figsize=(10*len(n_hidden_list), 10))
    fig_eigen, ax_eigen = plt.subplots(1,2)
    ev_per_dim = []
    cos_per_dim = []

    for i, n in enumerate(n_hidden_list):
        encoder = basemodel(n).to('cuda')
        checkpoint = torch.load(ckpt(n, ld))

        encoder.load_state_dict(checkpoint['model'])
        encoder.eval()
        print('Loaded {}th model'.format(i))

        ax = ax_bdry if len(n_hidden_list) == 1 else ax_bdry[i]
        draw_decision_countour(encoder, ax, use_adv, data_loader.X_train, data_loader.y_train, n_samples, n)
        print('Draw {}th contour'.format(i))

        whole_param = dict(encoder.named_parameters())
        hidden_L = whole_param['noise_L2.weight'].data.cpu().numpy()
        noise_hidden_cov = np.matmul(hidden_L, hidden_L.T)

        _, eigen_ld, eigen_v = np.linalg.svd(noise_hidden_cov)
        ev_per_dim.append(eigen_ld[ :2])

        _, h = encoder(X)
        mu1, mu2 = torch.mean(h[ :n_samples//2], dim=0), torch.mean(h[n_samples//2: ], dim=0)
        dmu = mu1 - mu2
        dmu = dmu.data.cpu().numpy()

        cos_sim = []
        for v in eigen_v[ :2]:
            cos = np.abs(np.dot(dmu, v)) / np.linalg.norm(dmu)
            cos_sim.append(cos)
        cos_per_dim.append(cos_sim)
        print('[{} neuron]: {}'.format(n, np.array(cos_per_dim).T))
    ev_per_dim = np.array(ev_per_dim).T
    cos_per_dim = np.array(cos_per_dim).T

    draw_eigen(ev_per_dim, cos_per_dim, ax_eigen, n_hidden_list)

    fig_bdry.savefig('log/gaussian_test/bdry_{}_{}to{}_adv_{}.png'.format(ld, n_hidden_list[0], n_hidden_list[-1], use_adv))
    fig_eigen.savefig('log/gaussian_test/eigen_{}_{}to{}_adv_{}.png'.format(ld, n_hidden_list[0], n_hidden_list[-1], use_adv))


if __name__ == '__main__':
    if args.eval:
        eval(args.n_hidden)

    else:
        train()


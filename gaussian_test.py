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
from gaussian.Visualize_ import draw_decision_countour

np.random.seed(3)
torch.manual_seed(5)


parser = argparse.ArgumentParser(description="Generalization by Noise")
parser.add_argument("--n_hidden", type=int, nargs='+', default=None, required=True,
                    help="number of hidden neurons.")
parser.add_argument("--eval", default=False, action='store_true', required=False,
                    help="eval mode.")
parser.add_argument("--gpu", type=int, nargs='+', default=None, required=True,
                    help="choose gpu device.")
parser.add_argument("--architecture", type=str, default='advGNI', required=True,
                    help="")
parser.add_argument("--epsilon", type=float, default=1., required=False,
                    help="")

args = parser.parse_args()


gpus_tobe_used = ','.join([str(gpuNum) for gpuNum in args.gpu])
print('gpus_tobe_used: {}'.format(gpus_tobe_used))
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpus_tobe_used)

cudnn.enabled = True
cudnn.benchmark = True

# ---------------------------
# Parameters
# ---------------------------

""" 1. Gaussian Data """
n_samples = 1000 # Per domain
data_noise = 0.2
data_format = 'uniform' # uniform, twomoon, circles

""" 2. Optimizer """
lr = 1e-3
epoch = 500

""" 3. Architecture """
architecture = args.architecture
assert architecture == 'advGNI' or architecture == 'GNI' or architecture == 'FGSM' or architecture == 'base'
epsilon = args.epsilon
n_hidden = args.n_hidden[0]
if args.eval:
    n_hidden = args.n_hidden

# ---------------------------------------------------------
data_loader = DataGen(n_samples, data_noise, data_format)

X = torch.tensor(data_loader.X_train).float().to('cuda')
Y = torch.tensor(data_loader.y_train).to('cuda')
X_test = torch.tensor(data_loader.X_test).float().to('cuda')
Y_test = torch.tensor(data_loader.y_test).to('cuda')
#cen=torch.nn.CrossEntropyLoss()
cen=torch.nn.BCEWithLogitsLoss()

def train():
    encoder = basemodel(n_hidden, architecture, epsilon=epsilon).to('cuda')

    opt_theta = torch.optim.SGD(encoder.parameters(), lr=lr, momentum=0.9)
    encoder.train()

    for e in range(epoch):
        if architecture == 'advGNI':
            # -------------------------
            # 1. Obtain a grad mask
            # -------------------------
            X.requires_grad = True
            logit_clean = encoder(X, hook=True)
            loss = cen(logit_clean, Y)

            loss.backward()
            grad = X.grad.clone().data
            encoder.grads['input'] = grad

            # -------------------------
            # 2. Train theta
            # -------------------------
            opt_theta.zero_grad()
            encoder.zero_grad()

            logit_adv = encoder(X, add_adv=True)

            # Main loss with adversarial example
            theta_loss = cen(logit_adv, Y)
            theta_loss.backward()
            opt_theta.step()

        elif architecture == 'FGSM':
            X.requires_grad = True
            logit_clean = encoder(X)
            loss = cen(logit_clean, Y)

            encoder.zero_grad()
            loss.backward()
            grad = X.grad.data
            encoder.grads['input'] = grad

            opt_theta.zero_grad()
            encoder.zero_grad()

            logit_adv = encoder(X, add_adv=True)
            loss = cen(logit_adv, Y)
            loss.backward()
            opt_theta.step()

        else:
            opt_theta.zero_grad()
            output = encoder(X)
            loss = cen(output, Y)
            loss.backward()
            opt_theta.step()

    encoder.eval()
    acc = validation(X_test, Y_test, encoder)
    print('[n_hidden: {}] Acc={}'.format(n_hidden, acc))

    torch.save({
        'model': encoder.state_dict(),
    }, os.path.join('snapshots', 'gaussian_test', '{}_{}.pth'.format(architecture, n_hidden)))
    print('taking snapshot ...')

def validation(x_test, y_test, encoder):
    logit = encoder(x_test)
    #pred = logit.data.max(1).indices
    pred = logit.data > 0.
    correct = pred.eq(y_test.data).cpu().sum()
    acc = 100.*correct/len(y_test)
    return acc

def eval(n_hidden_list):
    ckpt = lambda x,y: os.path.join('snapshots', 'gaussian_test', '{}_{}.pth'.format(x, y))

    fig_bdry, ax_bdry = plt.subplots(1, len(n_hidden_list), figsize=(10*len(n_hidden_list), 10))
    ev_per_dim = []
    cos_per_dim = []

    for i, n in enumerate(n_hidden_list):
        encoder = basemodel(n, architecture, epsilon=epsilon).to('cuda')
        checkpoint = torch.load(ckpt(architecture, n))

        encoder.load_state_dict(checkpoint['model'])
        encoder.eval()
        print('Loaded {}th model'.format(i))

        ax = ax_bdry if len(n_hidden_list) == 1 else ax_bdry[i]
        draw_decision_countour(encoder, ax, data_loader.X_train, data_loader.y_train, n_samples, n)
        print('Draw {}th contour'.format(i))

    fig_bdry.savefig('log/gaussian_test/bdry_{}_{}to{}.png'.format(architecture, n_hidden_list[0], n_hidden_list[-1]))


if __name__ == '__main__':
    if args.eval:
        eval(args.n_hidden)

    else:
        train()


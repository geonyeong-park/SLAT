import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
import os
import shutil
import sys
import pickle as pkl
from timeit import default_timer as timer
from math import sqrt
import numpy as np

import advertorch
from advertorch.attacks import LinfPGDAttack, L2PGDAttack
from advertorch.context import ctx_noparamgrad_and_eval
from model.resnet import PreActResNet18
from model.wide_resnet import WideResNet28_10
from model.utils import attack_pgd, std_t, clamp, lower_limit, upper_limit, cosine_similarity
from model.PGDtrainer import PGD
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from Visualize import plot_embedding


class GenByNoise(object):
    def __init__(self, dataset, config):
        self.config = config
        self.device = self._get_device()
        self.log_dir = config['exp_setting']['log_dir']
        self.snapshot_dir = config['exp_setting']['snapshot_dir']
        self.writer = SummaryWriter(log_dir=self.log_dir)

        self.dataset = dataset
        self.data_name = config['dataset']['name']
        self.num_cls = config['dataset'][self.data_name]['num_cls']
        self.batch_size = config['dataset'][self.data_name]['batch_size']
        self.input_size = config['dataset'][self.data_name]['input_size']
        self.structure = config['model']['baseline']

        network = config['model'][self.data_name]
        if network == 'ResNet':
            self.model = PreActResNet18(config).to(self.device)
            eta = config['model']['ResNet']['eta']
        elif network == 'Wide_ResNet':
            self.model = WideResNet28_10(config).to(self.device)
            eta = config['model']['ResNet']['eta']
        else:
            raise ValueError('Not implemented yet')

        self.model_checkpoints_folder = config['exp_setting']['snapshot_dir']
        self.train_loader, self.valid_loader = self.dataset.get_data_loaders()
        self.cen = nn.CrossEntropyLoss()
        self.epochs = self.config['train']['num_epochs']
        self._get_optimizer()

        self.log_loss = {}
        self.log_loss['val_acc'] = []
        self.log_loss['val_loss'] = []
        self.log_loss['val_cos'] = []
        self.log_loss['val_norm'] = []
        self.log_loss['training_time'] = 0.
        self.val_epoch = 10 if self.structure != 'Free' else 1

        self.tsne = TSNE(n_components=2, perplexity=20, init='pca', n_iter=3000)

        # For adversarial robustness test
        self.epsilon = (eta/ 255.) / std_t
        self.pgd_alpha = (eta*0.25 / 255.) / std_t

        if config['model']['baseline'] == 'CURE':
            checkpoint = torch.load('snapshots/ongoing_CURE_wideresnet/pretrain.pth')
            self.model.load_state_dict(checkpoint['model'])
            self.opt_theta = torch.optim.Adam(self.model.parameters(), lr=1e-4)
            self.theta_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.opt_theta, [60,120,160], 0.1)
            self.epochs = 50
            print('load ckpt for CURE')
        if config['model']['baseline'] == 'Free':
            self.replay = config['model']['Free']['replay']
            self.delta = torch.zeros(self.batch_size, 3, self.input_size, self.input_size).to('cuda')
            self.delta.requires_grad = True
            self.epochs = int(self.epochs/self.replay)
            step1, step2, step3 = int(self.epochs*0.3), int(self.epochs*0.6), int(self.epochs*0.8)
            self.theta_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.opt_theta, [step1,step2,step3], 0.1)

    def _get_device(self):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("Running on:", device)
        return device

    def _get_optimizer(self):
        opt_param = self.config['optimizer']

        self.opt_theta = torch.optim.SGD(self.model.parameters(), opt_param['lr'][self.data_name],
                                         weight_decay=opt_param['weight_decay'], momentum=opt_param['momentum'])
        step1, step2, step3 = int(self.epochs*0.3), int(self.epochs*0.6), int(self.epochs*0.8)
        self.theta_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.opt_theta, [step1,step2,step3], 0.1)

    def pretrain(self):
        """For CURE"""
        for e in range(30):
            self.model.train()

            for i, (x, y) in enumerate(self.train_loader):
                x = x.to(self.device)
                y = y.long().to(self.device)
                self.opt_theta.zero_grad()
                self.model.zero_grad()

                logit_clean = self.model(x)
                loss = self.cen(logit_clean, y)
                loss.backward()
                self.opt_theta.step()
            self._validation(e, None, advattack=False)
        torch.save({
            'model': self.model.state_dict(),
        }, os.path.join(self.snapshot_dir, 'pretrain.pth'))
        return

    def train(self):
        sample_path = os.path.join(self.log_dir, '{}epoch_log.pkl')
        n_iter = 0
        training_time = 0
        self.prev_adv_acc = 0.

        for e in range(self.epochs):
            start = timer()
            self.model.train()

            n_iter = self._train_epoch(e, n_iter)
            end = timer()
            training_time += end - start

            if (e+1) % self.val_epoch == 0:
                print('Training time: {}'.format(training_time))
                self._validation(e, n_iter, advattack=True)

        print('Training Finished.')
        print('taking snapshot ...')
        torch.save({
            'model': self.model.state_dict(),
        }, os.path.join(self.snapshot_dir, 'pretrain.pth'))

        with open(sample_path.format(e+1), 'wb') as f:
            self.log_loss['training_time'] = training_time
            pkl.dump(self.log_loss, f, pkl.HIGHEST_PROTOCOL)
        print('Finished, elapsed time: {}'.format(training_time))

    def _train_epoch(self, epoch, n_iter):
        for i, (x, y) in enumerate(self.train_loader):
            x = x.to(self.device)
            y = y.long().to(self.device)
            if i==0: first_batch = x,y
            self.opt_theta.zero_grad()
            self.model.zero_grad()

            if self.structure == 'advGNI' or self.structure == 'advGNI_GA':
                # -------------------------
                # 1. Obtain a grad mask
                # -------------------------
                x.requires_grad = True
                logit_clean = self.model(x, hook=True)
                loss = self.cen(logit_clean, y)

                loss.backward()
                grad = x.grad.clone().data
                self.model.grads['input'] = grad
                # -------------------------
                # 2. Train theta
                # -------------------------
                self.opt_theta.zero_grad()
                self.model.zero_grad()

                logit_adv = self.model(x, add_adv=True)

                # Main loss with adversarial example
                theta_loss = self.cen(logit_adv, y)
                retain_graph = True if self.structure == 'advGNI_GA' else False
                theta_loss.backward(retain_graph=retain_graph)

                if self.structure == 'advGNI_GA':
                    adv_grad = x.grad.data
                    cos = cosine_similarity(grad, adv_grad)
                    cos.requires_grad = True
                    if n_iter % 1000 == 0: print(cos)
                    reg = self.config['model']['FGSM_GA']['coeff']*(1. - cos)
                    reg.backward()

                self.opt_theta.step()

            elif self.structure == 'Free':
                for _ in range(self.replay):
                    output = self.model(x + self.delta[:x.size(0)])
                    loss = self.cen(output, y)
                    self.opt_theta.zero_grad()
                    loss.backward()

                    grad = self.delta.grad.detach()
                    self.delta.data = clamp(self.delta + self.epsilon * torch.sign(grad), -self.epsilon, self.epsilon)
                    self.delta.data[:x.size(0)] = clamp(self.delta[:x.size(0)], lower_limit-x, upper_limit-x)
                    self.opt_theta.step()
                    self.delta.grad.zero_()

            elif self.structure == 'CURE':
                x.requires_grad = True
                logit_clean = self.model(x)
                loss = self.cen(logit_clean, y)

                loss.backward()
                grad = x.grad.data
                z = torch.sign(grad).detach() + 0.
                h = min(1.5, 0.3*(epoch+1))
                z = h*z / z.view(z.shape[0], -1).norm(dim=1)[:,None,None,None]

                self.opt_theta.zero_grad()
                self.model.zero_grad()

                logit_hz = self.model(x+z)
                loss_hz = self.cen(logit_hz, y)
                grad_z = torch.autograd.grad(loss_hz, x, create_graph=True)[0]
                grad_diff = grad - grad_z

                reg = torch.mean(grad_diff.view(grad_diff.shape[0], -1).norm(dim=1))

                loss = self.cen(self.model(x), y)
                loss = loss + self.config['model']['CURE']['gamma']*reg
                loss.backward()
                self.opt_theta.step()

            elif self.structure == 'FGSM' or self.structure == 'FGSM_GA':
                x.requires_grad = True
                logit_clean = self.model(x)
                loss = self.cen(logit_clean, y)

                self.model.zero_grad()
                loss.backward()
                grad = x.grad.data
                self.model.grads['input'] = grad

                self.opt_theta.zero_grad()
                self.model.zero_grad()

                logit_adv = self.model(x, add_adv=True)
                loss = self.cen(logit_adv, y)

                reg = torch.zeros(1).cuda()[0]

                if self.structure == 'FGSM_GA':
                    # Gradient alignment
                    delta = torch.zeros(x.shape).cuda()
                    for j in range(len(self.epsilon)):
                        delta[:, j, :, :].uniform_(-self.epsilon[j][0][0].item(), self.epsilon[j][0][0].item())
                        delta.data = clamp(delta, lower_limit - x, upper_limit - x)
                    delta.requires_grad = True

                    delta_output = self.model(x + delta)
                    delta_loss = self.cen(delta_output, y)

                    adv_grad = torch.autograd.grad(delta_loss, delta, create_graph=True)[0]
                    cos = cosine_similarity(grad, adv_grad)
                    #grad, adv_grad = grad.reshape(len(grad), -1), adv_grad.reshape(len(adv_grad), -1)
                    #cos = torch.nn.functional.cosine_similarity(grad, adv_grad, 1)
                    if n_iter % 1000 == 0: print(cos)
                    reg = self.config['model']['FGSM_GA']['coeff']*(1. - cos)

                loss += reg
                loss.backward()
                self.opt_theta.step()

            elif self.structure == 'PGD':
                # Code is partially from
                # https://github.com/locuslab/fast_adversarial/blob/master/CIFAR10/train_pgd.py
                delta = torch.zeros_like(x).to('cuda')
                delta.requires_grad = True
                for i in range(self.config['model']['PGD']['iters']):
                    output = self.model(x + delta)
                    loss = self.cen(output, y)
                    loss.backward()

                    grad = delta.grad.detach()
                    delta.data = clamp(delta + self.pgd_alpha * torch.sign(grad), -self.epsilon, self.epsilon)
                    delta.data = clamp(delta, lower_limit - x, upper_limit - x)
                    delta.grad.zero_()
                delta = delta.detach()
                output = self.model(x + delta)
                loss = self.cen(output, y)
                self.opt_theta.zero_grad()
                loss.backward()
                self.opt_theta.step()

            elif self.structure == 'FGSM_RS':
                logit = self.model(x)
                loss = self.cen(logit, y)
                loss.backward()

                self.opt_theta.zero_grad()
                logit = self.model(x, add_adv=True)
                loss = self.cen(logit, y)
                loss.backward()
                self.opt_theta.step()

            else:
                self.opt_theta.zero_grad()
                loss = self._step(self.model, x, y)
                loss.backward()
                self.opt_theta.step()

            n_iter += 1

        self.theta_scheduler.step()

        x,y = first_batch

        self.model.eval()
        pgd_delta = attack_pgd(self.model, x, y, self.epsilon, self.pgd_alpha, 7, 1)
        with torch.no_grad():
            output = self.model(clamp(x + pgd_delta[:x.size(0)], lower_limit, upper_limit))
        robust_acc = (output.max(1)[1] == y).sum().item() / y.size(0)

        if robust_acc > self.prev_adv_acc:
            torch.save({
                'model': self.model.state_dict(),
            }, os.path.join(self.snapshot_dir, 'pretrain_best.pth'))

        print('[{} epoch, adv_test] acc={}'.format(epoch, robust_acc))
        self.prev_adv_acc = robust_acc
        self.model.train()
        return n_iter

    def _step(self, model, x, y):
        model.train()
        logit = model(x)
        loss = self.cen(logit, y)
        return loss

    def _validation(self, epoch, n_iter, record=True, advattack=False, attack_iters=7, restarts=1):
        self.model.eval()

        valid_loss = 0.
        valid_acc = 0.
        valid_cos = 0.
        valid_norm = {
            'input': 0.,
            'conv1': 0.,
            'block1': 0.,
            'block2': 0.,
            'block3': 0.
        }
        counter = 0

        for x, y in self.valid_loader:
            x = x.to(self.device)
            y = y.long().to(self.device)

            # Computing validation accuracy and loss
            if advattack:
                with torch.enable_grad():
                    pgd_delta = attack_pgd(self.model, x, y, self.epsilon, self.pgd_alpha, attack_iters, restarts)
                with torch.no_grad():
                    logit = self.model(clamp(x + pgd_delta[:x.size(0)], lower_limit, upper_limit))
            else:
                logit = self.model(x)
            loss = self.cen(logit, y)
            valid_loss += loss.data.cpu().numpy()

            pred = logit.data.max(1)[1]
            valid_acc += pred.eq(y.data).cpu().sum()

            k = y.data.size()[0]
            counter += k

            # Computing cosine similarity
            self.model.eval()
            x.requires_grad = True

            logit = self.model(x)
            loss_ = self.cen(logit, y)
            loss_.backward()
            grad = x.grad.data

            delta = torch.zeros(x.shape).cuda()
            for j in range(len(self.epsilon)):
                delta[:, j, :, :].uniform_(-self.epsilon[j][0][0].item(), self.epsilon[j][0][0].item())
                delta.data = clamp(delta, lower_limit - x, upper_limit - x)
            delta.requires_grad = True

            delta_output = self.model(x + delta)
            delta_loss = self.cen(delta_output, y)

            adv_grad = torch.autograd.grad(delta_loss, delta, create_graph=True)[0]
            cos = cosine_similarity(grad, adv_grad)
            valid_cos += cos.data.cpu().numpy()
            # ------------------------------------------------------------------
            # Computing ||grad||

            logit = self.model(x, hook=True)
            loss = self.cen(logit, y)
            loss.backward()
            for k in self.model.grads.keys():
                if not k == 'input': grad = self.model.grads[k]
                valid_norm[k] = torch.mean(grad.view(grad.shape[0], -1).norm(p=1, dim=1)).data.cpu().numpy()

        num_batch = len(self.valid_loader)
        valid_loss /= counter
        valid_acc /= counter
        valid_cos /= num_batch
        for k in self.model.grads.keys(): valid_norm[k] = valid_norm[k] / num_batch

        if advattack:
            self.log_loss['val_loss'].append(valid_loss)
            self.log_loss['val_acc'].append(valid_acc)
            self.log_loss['val_cos'].append(valid_cos)
            self.log_loss['val_norm'].append(valid_norm)

        if epoch == None: epoch='Test'
        print('[{}/{}/PGD attack {}]: Acc={:.5f}, Loss={:.3f}, Cos={:.2f}, LR={:.5f}'.format(epoch, n_iter, advattack,
                                                       valid_acc, valid_loss, valid_cos, self.theta_scheduler.get_lr()[0]))
        print(valid_norm)

    def test(self, checkpoint, mode):
        self.model.load_state_dict(checkpoint['model'])
        self.model.eval()
        self.model.to(self.device)
        advattack = True if mode == 'adv_attack' else False
        self._validation(None, None, False, advattack=advattack, attack_iters=50, restarts=10)
        print('Test finished')

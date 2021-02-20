import torch
import torch.nn as nn
from torch.autograd import Variable
#from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
from torch.optim.lr_scheduler import MultiStepLR, CyclicLR
import os
import shutil
import sys
import pickle as pkl
from timeit import default_timer as timer
from math import sqrt, ceil
import numpy as np

import advertorch
from model.resnet import PreActResNet18
from model.wide_resnet import WideResNet28_10
from model.hidden_module import std_t, lower_limit, upper_limit
from utils.attack import attack_pgd
from utils.utils import clamp, cos_by_uniform, cosine_similarity


class Solver(object):
    def __init__(self, dataset, config):
        self.config = config
        self.device = self._get_device()
        self.log_dir = config['exp_setting']['log_dir']
        self.snapshot_dir = config['exp_setting']['snapshot_dir']
        #self.writer = SummaryWriter(log_dir=self.log_dir)

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
        self.schedule = self.config['optimizer']['schedule']
        self.epochs = self.config['train']['num_epochs'][self.schedule]
        self.lr_milestone = self.config['optimizer']['lr_milestone']

        if self.structure == 'Free':
            self.replay = config['model']['Free']['replay']
            self.delta = torch.zeros(self.batch_size, 3, self.input_size, self.input_size).to('cuda')
            self.delta.requires_grad = True
            self.epochs = int(self.epochs/self.replay)

            epochs_full = self.config['train']['num_epochs'][self.schedule]
            self.lr_milestone = [ceil(x/epochs_full*self.epochs) for x in self.lr_milestone]

        self._get_optimizer()

        self.log_loss = {}
        self.log_loss['val_acc'] = []
        self.log_loss['val_loss'] = []
        self.log_loss['val_cos'] = []
        self.log_loss['val_norm'] = []
        self.log_loss['training_time'] = 0.
        self.val_epoch = 10 if self.structure != 'Free' else 1

        # For adversarial robustness test
        self.epsilon = (eta/ 255.) / std_t
        self.pgd_alpha = (eta*0.25 / 255.) / std_t


    def _get_device(self):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("Running on:", device)
        return device

    def _get_optimizer(self):
        opt_param = self.config['optimizer']

        if self.schedule == 'multistep':
            self.opt_theta = torch.optim.SGD(self.model.parameters(), opt_param['lr']['multistep'],
                                            weight_decay=opt_param['weight_decay'], momentum=opt_param['momentum'])
            self.theta_scheduler = MultiStepLR(self.opt_theta, self.lr_milestone, 0.1)
        elif self.schedule == 'cyclic':
            self.opt_theta = torch.optim.SGD(self.model.parameters(), opt_param['lr']['cyclic'],
                                            weight_decay=opt_param['weight_decay'], momentum=opt_param['momentum'])
            if self.structure != 'Free':
                lr_steps = self.epochs * len(self.train_loader)
            else:
                lr_steps = self.epochs * len(self.train_loader) * self.replay
            self.theta_scheduler = CyclicLR(self.opt_theta, 0., opt_param['lr']['cyclic'],
                                            step_size_up=lr_steps*2./5., step_size_down=lr_steps*3/5.)

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
            self._validation(e, advattack=False)
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

            if self.schedule == 'cyclic' or (e+1) % self.val_epoch == 0:
                print('Training time: {}'.format(training_time))
                self._validation(e, advattack=True)

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

                self.opt_theta.zero_grad()
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
                    adv_grad = x.grad.clone().data
                    cos = cosine_similarity(grad, adv_grad)
                    cos.requires_grad = True
                    if n_iter % 1000 == 0: print(cos)
                    reg = self.config['model']['FGSM_GA']['coeff']*(1. - cos)
                    reg.backward()

                self.opt_theta.step()
                if self.schedule == 'cyclic': self.theta_scheduler.step()

            elif self.structure == 'CURE':
                h = 0.01
                x.requires_grad = True
                logit_clean = self.model(x)
                loss = self.cen(logit_clean, y)
                loss.backward()

                grad = x.grad.clone().data
                shape = grad.shape
                grad_ = grad.view(shape[0], -1)
                norm_grad = grad_.norm(2, dim=-1, keepdim=True)
                nonzero = norm_grad.view(-1)>0
                grad_[nonzero] = grad_[nonzero].div(norm_grad[nonzero])

                grad_ = grad_.view(shape)
                x_hgrad = x + h*grad_
                logit_h = self.model(x_hgrad)
                loss_h = self.cen(logit_h, y)

                reg = (loss - loss_h) / h
                reg = reg.pow(2).mean()/2.
                theta_loss = loss + 0.1*reg

                self.opt_theta.zero_grad()
                theta_loss.backward()
                self.opt_theta.step()
                if self.schedule == 'cyclic': self.theta_scheduler.step()

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
                    if self.schedule == 'cyclic': self.theta_scheduler.step()

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

                if 'FGSM_GA' in self.structure:
                    # Gradient alignment
                    cos_ = cos_by_uniform(self.model, x, y, self.epsilon, grad)
                    if n_iter % 1000 == 0: print(cos_)
                    reg = self.config['model']['FGSM_GA']['coeff']*(1. - cos_)

                loss += reg
                loss.backward()
                self.opt_theta.step()
                if self.schedule == 'cyclic': self.theta_scheduler.step()

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
                if self.schedule == 'cyclic': self.theta_scheduler.step()

            elif self.structure == 'FGSM_RS':
                logit = self.model(x)
                loss = self.cen(logit, y)
                loss.backward()

                self.opt_theta.zero_grad()
                logit = self.model(x, add_adv=True)
                loss = self.cen(logit, y)
                loss.backward()
                self.opt_theta.step()
                if self.schedule == 'cyclic': self.theta_scheduler.step()

            else:
                self.opt_theta.zero_grad()
                logit_clean = self.model(x)
                loss = self.cen(logit_clean, y)
                loss.backward()
                self.opt_theta.step()
                if self.schedule == 'cyclic': self.theta_scheduler.step()

            n_iter += 1

        if self.schedule == 'multistep': self.theta_scheduler.step()

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

    def _validation(self, epoch, advattack=False, attack_iters=7, restarts=1):
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

            x.requires_grad = True
            logit = self.model(x)
            loss_ = self.cen(logit, y)
            loss_.backward()
            grad = x.grad.data

            cos = cos_by_uniform(self.model, x, y, self.epsilon, grad)
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
        print('[{}/PGD attack {}]: Acc={:.5f}, Loss={:.3f}, Cos={:.2f}, LR={:.5f}'.format(epoch, advattack,
                                                       valid_acc, valid_loss, valid_cos, self.theta_scheduler.get_lr()[0]))
        print(valid_norm)

    def test(self, checkpoint):
        self.model.load_state_dict(checkpoint['model'])
        self.model.eval()
        self.model.to(self.device)
        self._validation(None, advattack=True, attack_iters=50, restarts=10)
        print('Test finished')

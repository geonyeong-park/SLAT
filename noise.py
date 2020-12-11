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
from model.utils import attack_pgd, std_t, clamp, lower_limit, upper_limit
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
        self.structure = config['model']['baseline']

        network = config['model'][self.data_name]
        if network == 'ResNet':
            self.model = PreActResNet18(config).to(self.device)
        else:
            raise ValueError('Not implemented yet')

        self.model_checkpoints_folder = config['exp_setting']['snapshot_dir']
        self.train_loader, self.valid_loader = self.dataset.get_data_loaders()
        self.cen = nn.CrossEntropyLoss()
        self.using_adv_noise = self._get_optimizer()
        print('Using adversarial noise distribution: ', self.using_adv_noise)

        self.log_loss = {}
        self.log_loss['train_loss'] = []
        self.log_loss['train_KL'] = []
        self.log_loss['val_acc'] = []
        self.log_loss['val_loss'] = []
        self.log_loss['training_time'] = 0.
        self.ld = config['train']['ld'][self.data_name]
        self.val_epoch = 1

        self.tsne = TSNE(n_components=2, perplexity=20, init='pca', n_iter=3000)
        # For adversarial robustness test
        self.epsilon = (8/ 255.) / std_t
        self.pgd_alpha = (2 / 255.) / std_t

    def _get_device(self):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("Running on:", device)
        return device

    def _get_optimizer(self):
        opt_param = self.config['optimizer']
        epochs = self.config['train']['num_epochs']

        lr_steps = epochs * len(self.train_loader)

        self.opt_theta = torch.optim.SGD(self.model.parameters(), opt_param['lr'][self.data_name],
                                         weight_decay=opt_param['weight_decay'], momentum=opt_param['momentum'])
        self.theta_scheduler = torch.optim.lr_scheduler.CyclicLR(self.opt_theta, base_lr=0, max_lr=opt_param['lr'][self.data_name],
                                                      step_size_up=lr_steps / 2, step_size_down=lr_steps / 2)

        use_adversarial_noise = self.model.use_adversarial_noise
        return use_adversarial_noise

    def train(self):
        epochs = self.config['train']['num_epochs']
        sample_path = os.path.join(self.log_dir, '{}epoch_log.pkl')
        n_iter = 0
        training_time = 0
        self.prev_adv_acc = 0.

        for e in range(epochs):
            start = timer()
            self.model.train()

            n_iter = self._train_epoch(e, n_iter)
            end = timer()
            training_time += end - start

            if (e+1) % self.val_epoch == 0:
                self.model.eval()
                self._validation(e, n_iter, advattack=False)

            if n_iter % self.config['exp_setting']['log_every_n_steps'] == 0:
                self.writer.add_scalar('lr/theta', self.theta_scheduler.get_lr()[0], global_step=n_iter)

            if (e+1) % self.config['exp_setting']['save_every_n_epochs'] == 0:
                print('taking snapshot ...')
                torch.save({
                    'model': self.model.state_dict(),
                }, os.path.join(self.snapshot_dir, 'pretrain_'+str(e+1)+'.pth'))

                with open(sample_path.format(e+1), 'wb') as f:
                    self.log_loss['training_time'] = training_time
                    pkl.dump(self.log_loss, f, pkl.HIGHEST_PROTOCOL)

        print('Training Finished, elapsed time: {}'.format(training_time))

    def _train_epoch(self, epoch, n_iter):
        for i, (x, y) in enumerate(self.train_loader):
            x = x.to(self.device)
            y = y.long().to(self.device)
            if i==0: first_batch = x,y
            x,y = self._preprocess(x,y)

            if self.using_adv_noise:
                # -------------------------
                # 1. Obtain a grad mask
                # -------------------------
                grad_mask = []
                x.requires_grad = True
                logit_clean = self.model(x)
                loss = self.cen(logit_clean, y)

                loss.backward()
                grad_mask.append(x.grad.data)

                # -------------------------
                # 2. Generate adversarial example
                # -------------------------
                self.opt_theta.zero_grad()
                self.model.zero_grad()

                logit_adv = self.model(x, grad_mask, add_adv=True)
                yhat_noise = nn.Softmax(dim=1)(logit_adv)

                # Main loss with adversarial example
                adv_loss = self.cen(logit_adv, y)

                # Regularize KL-divergence for approximated second-order penalty
                logit_clean = self.model(x)
                yhat_clean = nn.Softmax(dim=1)(logit_clean)
                KL = nn.KLDivLoss(reduction='batchmean')(yhat_clean.log(), yhat_noise)

                loss = adv_loss + self.ld * KL
                loss.backward()
                self.opt_theta.step()

            elif self.structure == 'FGSM':
                grad_mask = []
                x.requires_grad = True
                logit = self.model(x)
                loss = self.cen(logit, y)

                self.model.zero_grad()
                loss.backward()
                grad_mask.append(x.grad.data)

                self.opt_theta.zero_grad()
                self.model.zero_grad()

                logit = self.model(x, grad_mask, add_adv=True)
                loss = self.cen(logit, y)
                loss.backward()
                self.opt_theta.step()

            else:
                self.opt_theta.zero_grad()
                loss = self._step(self.model, x, y)
                loss.backward()
                self.opt_theta.step()

            if n_iter % self.config['exp_setting']['log_every_n_steps'] == 0:
                self.writer.add_scalar('train/loss', loss, global_step=n_iter)
                self.log_loss['train_loss'].append(loss.data.cpu().numpy())
                if self.using_adv_noise:
                    self.log_loss['train_KL'].append(KL.data.cpu().numpy())

            n_iter += 1
            self.theta_scheduler.step()

        x,y = first_batch

        pgd_delta = attack_pgd(self.model, x, y, self.epsilon, self.pgd_alpha, 5, 1)
        with torch.no_grad():
            output = self.model(clamp(x + pgd_delta[:x.size(0)], lower_limit, upper_limit))
        robust_acc = (output.max(1)[1] == y).sum().item() / y.size(0)
        if robust_acc - self.prev_adv_acc < -0.2:
            torch.save({
                'model': self.model.state_dict(),
            }, os.path.join(self.snapshot_dir, 'pretrain_'+str(epoch)+'.pth'))
            raise Warning('Erroneous behavior, epoch={}, \
                          acc={}, prev_adv_acc={}'.format(epoch, robust_acc, self.prev_adv_acc))
        print('[{} epoch, adv_test] acc={}'.format(epoch, robust_acc))
        self.prev_adv_acc = robust_acc

        return n_iter

    def _step(self, model, x, y):
        model.train()
        logit = model(x)

        loss = self._mix_loss(logit,y) if self.structure=='mixup' else self.cen(logit, y)
        return loss

    def _mix_loss(self, logit, y):
        y_a, y_b, lam = y[0], y[1], y[2]
        return lam * self.cen(logit, y_a) + (1 - lam) * self.cen(logit, y_b)

    def _preprocess(self, x, y):
        if self.structure == 'mixup':
            alpha = self.config['model']['mixup']['alpha']
            lam = np.random.beta(alpha, alpha)

            batch_size = x.size()[0]
            index = torch.randperm(batch_size).cuda()

            mixed_x = lam * x + (1 - lam) * x[index, :]
            y_a, y_b = y, y[index]
            return mixed_x, (y_a, y_b, lam)
        else:
            return x, y

    def _validation(self, epoch, n_iter, record=True, advattack=False):
        with torch.no_grad():
            self.model.eval()

            valid_loss = 0.
            valid_acc = 0.
            counter = 0

            for x, y in self.valid_loader:
                x = x.to(self.device)
                y = y.long().to(self.device)

                if advattack:
                    with torch.no_grad():
                        with torch.enable_grad():
                            x = self.adversary.perturb(x, y)

                logit = self.model(x)
                loss = self.cen(logit, y)
                valid_loss += loss.data.cpu().numpy()

                pred = logit.data.max(1)[1]
                valid_acc += pred.eq(y.data).cpu().sum()

                k = y.data.size()[0]
                counter += k

            valid_loss /= counter
            valid_acc /= counter

            if record:
                self.writer.add_scalar('valid/loss', valid_loss, global_step=n_iter)
                self.writer.add_scalar('valid/acc', valid_acc, global_step=n_iter)
            self.log_loss['val_loss'].append(valid_loss)
            self.log_loss['val_acc'].append(valid_acc)

        if epoch == None: epoch='Test'
        print('[{}/{}]: Acc={:.3f}, Loss={:.3f}, LR={:.5f}'.format(epoch, n_iter,
                                                       valid_acc, valid_loss, self.theta_scheduler.get_lr()[0]))

    def test(self, checkpoint, mode):
        self.model.load_state_dict(checkpoint['model'])
        self.model.eval()
        self._PCA_tSNE()
        self.model.to(self.device)
        advattack = True if mode == 'adv_attack' else False
        self._validation(None, None, False, advattack=advattack)
        print('Test finished')

    def _PCA_tSNE(self):
        image, label = next(iter(self.valid_loader))
        sample_path = lambda x: os.path.join(self.log_dir, '{}.jpg'.format(x))
        logit = self.model.cpu()(image.cpu())
        tsne = self.tsne.fit_transform(logit.data.cpu().numpy())
        plot_embedding(tsne, label.data.cpu().numpy(), sample_path('tSNE'))

        pca = PCA(n_components=2)
        pc = pca.fit_transform(logit.data.cpu().numpy())
        plot_embedding(pc, label.data.cpu().numpy(), sample_path('PCA'))
        print('Plot tSNE and PCA results')

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

import advertorch
from advertorch.attacks import LinfPGDAttack
from advertorch.context import ctx_noparamgrad_and_eval
from model.FC import NoisyFC, NormalFC, Wrapper
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

        structure = config['model'][self.data_name]
        assert structure == 'FC' or structure =='NormalFC'
        if structure == 'FC':
            self.model = NoisyFC(config).to(self.device)
        elif structure == 'NormalFC':
            self.model = NormalFC(config).to(self.device)

        self.model_checkpoints_folder = config['exp_setting']['snapshot_dir']
        self.train_loader, self.valid_loader = self.dataset.get_data_loaders()
        self.cen = nn.CrossEntropyLoss()
        self.using_noise = self._get_optimizer()
        print('Using adversarial noise distribution: ', self.using_noise)
        self.using_advertorch = config['model']['adv_train']

        self.log_loss = {}
        self.log_loss['train_theta_loss'] = []
        self.log_loss['train_noise_loss'] = []
        self.log_loss['val_acc'] = []
        self.log_loss['val_loss'] = []
        self.log_loss['val_KL'] = []
        self.log_loss['training_time'] = 0.
        self.ld = config['train']['ld']
        self.val_epoch = 1

        self.tsne = TSNE(n_components=2, perplexity=20, init='pca', n_iter=3000)

        # In case for testing adversarial vulnerability or do training
        self.adversary = LinfPGDAttack(
            self.model, loss_fn=nn.CrossEntropyLoss(), eps=0.3,
            nb_iter=7, eps_iter=0.01, rand_init=True, clip_min=-2.0, clip_max=2.0,
            targeted=False)

    def _get_device(self):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("Running on:", device)
        return device

    def _get_optimizer(self):
        opt_param = self.config['optimizer']
        whole_param = dict(self.model.named_parameters())
        whole_param_keys = whole_param.keys()
        noise_param = [whole_param[key] for key in whole_param.keys() if 'noise' in key]
        theta_param = [whole_param[key] for key in whole_param.keys() if not 'noise' in key]

        self.opt_theta = torch.optim.SGD(theta_param, opt_param['lr'],
                                         weight_decay=opt_param['weight_decay'], momentum=opt_param['momentum'])
        self.opt_noise = torch.optim.SGD(noise_param, opt_param['lr'],
                                         weight_decay=opt_param['weight_decay'], momentum=opt_param['momentum'])
        self.theta_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.opt_theta, gamma=opt_param['lr_decay'])
        self.noise_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.opt_noise, gamma=opt_param['lr_decay'])

        use_adversarial_noise = self.model.use_adversarial_noise
        return use_adversarial_noise

    def train(self):
        epochs = self.config['train']['num_epochs']
        sample_path = os.path.join(self.log_dir, '{}epoch_log.pkl')
        n_iter = 0
        training_time = 0

        for e in range(epochs):
            start = timer()
            self.model.train()

            n_iter = self._train_epoch(e, n_iter)
            end = timer()
            training_time += end - start

            if (e+1) % self.val_epoch == 0:
                self.model.eval()
                self._validation(e, n_iter, advattack=False)

            if e >= 10:
                self.theta_scheduler.step()
                self.noise_scheduler.step()

            if n_iter % self.config['exp_setting']['log_every_n_steps'] == 0:
                self.writer.add_scalar('lr/theta', self.theta_scheduler.get_lr()[0], global_step=n_iter)
                self.writer.add_scalar('lr/noise', self.noise_scheduler.get_lr()[0], global_step=n_iter)

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
        for x, y in self.train_loader:
            x = x.to(self.device)
            y = y.long().to(self.device)

            if self.using_advertorch:
                self.model.eval()
                with torch.no_grad():
                    with torch.enable_grad():
                        x = self.adversary.perturb(x, y)
                self.model.train()

            # -------------------------
            # Update theta
            self.opt_theta.zero_grad()
            theta_loss = self._step(self.model, x, y, 'theta')
            theta_loss.backward()
            self.opt_theta.step()
            # -------------------------
            # Update noise
            if self.using_noise:
                self.opt_noise.zero_grad()
                noise_loss = self._step(self.model, x, y, 'noise')
                noise_loss.backward()
                self.opt_noise.step()
            # -------------------------

            if n_iter % self.config['exp_setting']['log_every_n_steps'] == 0:
                self.writer.add_scalar('train_loss/theta', theta_loss, global_step=n_iter)
                self.log_loss['train_theta_loss'].append(theta_loss.data.cpu().numpy())
                if self.using_noise:
                    self.writer.add_scalar('train_loss/noise', noise_loss, global_step=n_iter)
                    self.log_loss['train_noise_loss'].append(noise_loss.data.cpu().numpy())

            n_iter += 1

        adv_noise_input = self.model.noisy_module[0].noise_layer(x)
        inf_norm = torch.mean(torch.norm(adv_noise_input, float('inf'), dim=1))
        fro_norm = torch.mean(torch.norm(adv_noise_input, float(2), dim=1))
        print('[{}] epoch || inf norm: {}, l2 norm: {}'.format(epoch, inf_norm, fro_norm))
        return n_iter

    def _step(self, model, x, y, update='theta'):
        model.train()
        logit = model(x)

        if update == 'theta':
            loss = self.cen(logit, y)
        elif update == 'noise':
            loss = -self.cen(logit, y) + self.ld * model.norm_penalty

        return loss

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

                loss = self._step(self.model, x, y, 'theta')
                valid_loss += loss.data.cpu().numpy()

                output = self.model(x)
                pred = output.data.max(1)[1]
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

    def _KL_div(self):
        _, self.clean_valid_loader = self.dataset.get_data_loaders()
        with torch.no_grad():
            self.model.eval()

            KL_diff = 0.
            counter = 0

            for (x_noise, _), (x_clean, _) in zip(self.valid_loader, self.clean_valid_loader):
                x_noise = x_noise.to(self.device)
                x_clean = x_clean.to(self.device)

                logit_noise, _ = self.model(x_noise)
                logit_clean, _ = self.model(x_clean)

                yhat_noise = nn.Softmax(dim=1)(logit_noise)
                yhat_clean = nn.Softmax(dim=1)(logit_clean)

                KL_diff += nn.KLDivLoss(reduction='batchmean')(yhat_clean.log(), yhat_noise)
                counter += x_noise.data.size()[0]

            KL_diff /= counter
            self.log_loss['val_KL'].append(KL_diff)
            print('KL difference between clean and noisy data: {}'.format(KL_diff))

    def _test_gaussian(self):
        var_noise_in_data = [0., 0.2, 0.4, 0.6, 0.8, 1.0]
        for var in var_noise_in_data:
            _, self.valid_loader = self.dataset.get_data_loaders(var)
            print('[Test] Variance of input noise={}'.format(var))
            self._validation(None, None, False)
            self._KL_div()

        with open(os.path.join(self.log_dir, 'Robust_to_noise_test.pkl'), 'wb') as f:
            pkl.dump(self.log_loss, f, pkl.HIGHEST_PROTOCOL)

    def _test_adv_vulnerability(self):
        self._validation(None, None, False, advattack=True)

    def test(self, checkpoint, mode):
        self.model.load_state_dict(checkpoint['model'])
        self.model.eval()
        self._PCA_tSNE()
        self.model.to(self.device)

        assert mode == 'gaussian' or mode == 'adv_attack'
        if mode == 'gaussian':
            self._test_gaussian()
        elif mode == 'adv_attack':
            self._test_adv_vulnerability()

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



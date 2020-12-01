from __future__ import division

import scipy.misc
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")
import pandas as pd
import imageio
import os
from skimage import img_as_ubyte
import pickle as pkl
import argparse

def plot_embedding(X, y, save_path, title=None):
    """Plot an embedding X with the class label y colored by the domain d."""
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)
    num_color = np.max(y) + 1
    cmap = plt.cm.get_cmap('rainbow', num_color)

    # Plot colors numbers
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111)

    for i in range(X.shape[0]):
        # plot colored number
        plt.scatter(X[i, 0], X[i, 1], color=cmap(y[i]), s=3)

    plt.xticks([]), plt.yticks([])
    if title is not None:
        plt.title(title)

    fig.savefig(save_path)
    plt.close('all')


def plot_acc(dataset, ylim):
    log_path = lambda x: 'log/{}/Robust_to_noise_test.pkl'.format(x)

    with open(log_path('MNIST_test'), 'rb') as f:
        adv_noise = pkl.load(f)
    with open(log_path('MNIST_normal_noiseO'), 'rb') as f:
        gauss_noise = pkl.load(f)
    with open(log_path('MNIST_normal_noiseX'), 'rb') as f:
        no_noise = pkl.load(f)

    acc = [adv_noise['val_acc'], gauss_noise['val_acc'], no_noise['val_acc']]

    step = [0., 0.2, 0.4, 0.6, 0.8, 1.0]
    color=['salmon', 'green','royalblue']
    for i, data in enumerate(acc):
        plt.plot(step, np.array([x.data.cpu().numpy() for x in data]), color=color[i], linestyle='-.')

    plt.xlabel('Variance of input noise')
    plt.ylabel('Accuracy (%)')
    plt.xticks(step, [str(i) for i in step])
    plt.ylim([ylim,None])
    plt.minorticks_on()
    plt.grid(True, linestyle='--')
    plt.title('{}'.format(dataset))
    plt.legend(['Adversarial Noise', 'Gaussian Noise', 'Baseline'])
    plt.savefig('{}_acc.png'.format(dataset))


def plot_bar(dataset):
    log_path = lambda x: 'log/{}/Robust_to_noise_test.pkl'.format(x)

    with open(log_path('MNIST_test'), 'rb') as f:
        adv_noise = pkl.load(f)
    with open(log_path('MNIST_normal_noiseX'), 'rb') as f:
        no_noise = pkl.load(f)

    var = np.array([0.2, 0.4, 0.6, 0.8, 1.0])

    adv_noise_KL = np.array([x.data.cpu().numpy() for x in adv_noise['val_KL'][1:]])
    no_noise_KL = np.array([x.data.cpu().numpy() for x in no_noise['val_KL'][1:]])

    fig, ax = plt.subplots(1,1)
    width = 0.09

    bar = ax.bar(var - width/2, adv_noise_KL, width, label='Adversarial Noise', color='salmon')
    #present_height(ax, bar)
    bar = ax.bar(var + width/2, no_noise_KL, width, label='Baseline', color='royalblue')
    #present_height(ax, bar)

    ax.set_xticks(var)
    ax.set_xticklabels([str(x) for x in var])

    dataset_title = dataset
    ax.set_xlabel('Variance of input noise')

    ax.yaxis.set_tick_params()
    ax.set_ylabel('KL-divergence')
    ax.legend(loc='upper left', shadow=True, ncol=1)

    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color='gray', linestyle='dashed', linewidth=0.5)

    plt.tight_layout()
    plt.savefig('{}_KL.png'.format(dataset), format='png', dpi=300)
    plt.show()

def present_height(ax, bar):
    for rect in bar:
        height = rect.get_height()
        posx = rect.get_x()+rect.get_width()*0.5
        posy = height*1.01
        ax.text(posx, posy, '%.3f' % height, rotation=90, ha='center', va='bottom')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--domain',type=str)
    parser.add_argument('--ylim',type=int,default=None)
    args = parser.parse_args()
    plot_acc(args.domain,args.ylim)
    plot_bar(args.domain)

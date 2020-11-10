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


def plot_figure(dataset, ylim):
    with open('Ablation_Align_{}/50000_log.pkl'.format(dataset), 'rb') as f:
        align = pkl.load(f)
    with open('Ablation_noAlign_{}/50000_log.pkl'.format(dataset), 'rb') as f:
        no_align = pkl.load(f)
    with open('Ablation_vanila_{}/50000_log.pkl'.format(dataset), 'rb') as f:
        vanila = pkl.load(f)
    with open('Ablation_multiD_{}/50000_log.pkl'.format(dataset), 'rb') as f:
        multiD = pkl.load(f)
    """
    with open('{}_50000_log.pkl'.format(dataset.lower()), 'rb') as f:
        mdan = pkl.load(f)
        mdan = [x*100 for x in mdan]
    with open('Ablation/Ablation_GRL_Adam_{}/50000_log.pkl'.format(dataset), 'rb') as f:
        grl = pkl.load(f)
    with open('Ablation/Ablation_sourceonly_{}/25000_log.pkl'.format(dataset), 'rb') as f:
        sourceonly = pkl.load(f)
    with open('Ablation/{}_50000_log.pkl'.format(dataset.lower()), 'rb') as f:
        mdan = pkl.load(f)
    """

    acc = [align['target_acc'], no_align['target_acc'], vanila['target_acc'], multiD['target_acc']]

    step = [1000*(i+1) for i in range(50)]
    color=['salmon', 'chocolate','darkorange', 'royalblue']
    for i, data in enumerate(acc):
        print(len(data))
        if i != len(acc)-1 and i != len(acc)-2:
            idx = np.array([i*10 for i in range(len(data) // 10)]).astype(int)
            plt.plot(step, np.array(data)[idx], color=color[i], linestyle='-.')
        else:
            plt.plot(step, np.array(data), color=color[i], linestyle='-.')

    plt.xlabel('Number of Steps (x1e3)')
    plt.ylabel('Accuracy (%)')
    plt.xticks([10000, 20000, 30000, 40000, 50000], ['10','20','30','40','50'])
    plt.ylim([ylim,None])
    plt.minorticks_on()
    plt.grid(True, linestyle='--')
    dataset_title = 'MNIST-M' if dataset == 'MNIST' else dataset
    plt.title('{}'.format(dataset_title))
    plt.legend(['MIAN', 'MIAN(No S-S align)', 'MIAN(No LS)', 'Multi_D'])
    plt.savefig('{}.png'.format(dataset))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--domain',type=str)
    parser.add_argument('--ylim',type=int,default=None)
    args = parser.parse_args()
    plot_figure(args.domain,args.ylim)

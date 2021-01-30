import scipy.misc
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LightSource
from matplotlib import cm
import pickle as pkl
import argparse
import torch
import torchvision
import torch.nn as nn
from utils.utils import clamp, lower_limit, upper_limit, std_t
from utils.attack import attack_FGSM, attack_pgd

def compute_perturb(model, image, label, vec_x, vec_y, range_x, range_y,
                 grid_size=50, loss=nn.CrossEntropyLoss(reduction='none'),
                 batch_size=128):
    rx = np.linspace(*range_x, grid_size)
    ry = np.linspace(*range_y, grid_size)

    images = []
    loss_list = []

    image = image.to('cuda')
    label = label.to('cuda')
    vec_x = vec_x.to('cuda')
    vec_y = vec_y.to('cuda')

    for j in ry :
        for i in rx :
            images.append(image + i*vec_x + j*vec_y)

            if len(images) == batch_size :
                images = torch.stack(images)
                labels = torch.stack([label]*batch_size)
                outputs = model(images)
                loss_list.append(loss(outputs, labels).data.cpu().numpy())
                images = []

    images = torch.stack(images)
    labels = torch.stack([label]*len(images))
    outputs = model(images)
    loss_list.append(loss(outputs, labels).data.cpu().numpy())
    loss_list = np.concatenate(loss_list).reshape(len(rx), len(ry))

    return rx, ry, loss_list

def plot_perturb_plt(rx, ry, zs, save_path, eps,
                     title=None, width=8, height=7, linewidth = 0.1,
                     pane_color=(0.0, 0.0, 0.0, 0.01),
                     tick_pad_x=0, tick_pad_y=0, tick_pad_z=1.5,
                     xlabel=None, ylabel=None, zlabel=None,
                     xlabel_rotation=0, ylabel_rotation=0, zlabel_rotation=0,
                     view_azimuth=230, view_altitude=30,
                     light_azimuth=315, light_altitude=45, light_exag=0) :

    xs, ys = np.meshgrid(rx, ry)

    fig = plt.figure(figsize=(width, height))
    ax = fig.add_subplot(111, projection='3d')

    if title is not None :
        ax.set_title(title)

    # The azimuth (0-360, degrees clockwise from North) of the light source. Defaults to 315 degrees (from the northwest).
    # The altitude (0-90, degrees up from horizontal) of the light source. Defaults to 45 degrees from horizontal.

    ls = LightSource(azdeg=light_azimuth, altdeg=light_altitude)
    cmap = plt.get_cmap('coolwarm')
    fcolors = ls.shade(zs, cmap=cmap, vert_exag=light_exag, blend_mode='soft')
    surf = ax.plot_surface(xs, ys, zs, rstride=1, cstride=1, facecolors=fcolors,
                           linewidth=linewidth, antialiased=True, shade=False, alpha=0.7)
    contour = ax.contourf(xs, ys, zs, zdir='z', offset=np.min(zs), cmap=cmap)

    #surf.set_edgecolor(edge_color)
    ax.view_init(azim=view_azimuth, elev=view_altitude)

    ax.xaxis.set_rotate_label(False)
    ax.yaxis.set_rotate_label(False)
    ax.zaxis.set_rotate_label(False)

    if xlabel is not None :
        ax.set_xlabel(xlabel, rotation=xlabel_rotation)
    if ylabel is not None :
        ax.set_ylabel(ylabel, rotation=ylabel_rotation)
    if zlabel is not None :
        ax.set_zlabel(zlabel, rotation=zlabel_rotation)

    x_min, x_max = xs[0][0], xs[0][-1]
    xtick_step = np.linspace(x_min, x_max, 5)
    y_min, y_max = ys[0][0], ys[-1][-1]
    ytick_step = np.linspace(y_min, y_max, 5)

    ax.set_xticks(xtick_step)
    ax.set_xticklabels(['{}'.format(int(eps*i)) for i in xtick_step])
    ax.set_yticks(ytick_step)
    ax.set_yticklabels(['{}'.format(int(eps*i)) for i in ytick_step])
    #ax.set_zticks(None)

    ax.xaxis.set_pane_color(pane_color)
    ax.yaxis.set_pane_color(pane_color)
    ax.zaxis.set_pane_color(pane_color)

    ax.tick_params(axis='x', pad=tick_pad_x)
    ax.tick_params(axis='y', pad=tick_pad_y)
    ax.tick_params(axis='z', pad=tick_pad_z)

    plt.savefig(save_path('loss_landscape'), format='png', dpi=300)
    print('saved loss landscape')

def visualize_perturb(model, x, y, epsilon, step, iter, path):
    adv_l2 = attack_l2(model, x, y, epsilon, step, iter)
    delta_norm = adv_l2.view(adv_l2.shape[0], -1).norm(p=2, dim=1).data.cpu().numpy()
    save_img(adv_l2, path('perturbed'))
    save_img(x, path('original'))

    print('mean ||delta||={}'.format(np.mean(delta_norm)))
    return

def attack_l2(model, x, y, epsilon, step, iter):
    delta = torch.zeros_like(x).to('cuda')
    delta.requires_grad = True
    for i in range(iter):
        output = model(x + delta)
        loss = nn.CrossEntropyLoss()(output, y)
        loss.backward()

        grad = delta.grad.detach()
        g_norm = grad.view(grad.shape[0], -1).norm(p=2,dim=1).detach()
        grad = grad / g_norm.view(-1,1,1,1)

        delta.data = clamp(delta + step*grad, lower_limit - x, upper_limit - x)
        delta.data = clamp_l2_norm(delta, epsilon)
        delta.grad.zero_()
    delta = delta.detach()
    #adversary = x+delta
    return delta

def clamp_l2_norm(delta, epsilon):
    norm_delta = delta.view(delta.shape[0], -1).norm(p=2,dim=1).detach()
    norm_clamped = torch.clamp(norm_delta, max=epsilon)
    delta_ = delta / norm_delta.view(-1,1,1,1) * norm_clamped.view(-1,1,1,1)
    return delta_


def save_img(tensor, path, ncols=8):
    img = torchvision.utils.make_grid(tensor.detach().cpu(), nrow=ncols, normalize=True)
    torchvision.utils.save_image(img, path)

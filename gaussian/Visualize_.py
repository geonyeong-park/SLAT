import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms


def draw_decision_countour(encoder, ax, use_adv, X, Y, n_samples, n):
    whole_param = dict(encoder.named_parameters())
    noise_input_param = whole_param['noise_L1.weight'].data.cpu().numpy()

    if use_adv:
        noise_input_cov = np.matmul(noise_input_param, noise_input_param.T)
    else:
        noise_input_cov = np.array([[0.1,0.], [0.,0.1]])

    xi = np.linspace(-2., 2., 100)
    yi = np.linspace(-2., 2., 100)
    Xmesh, Ymesh = np.meshgrid(xi, yi)

    show_domain(X, Y, ax, n_samples)
    plot_countour(noise_input_cov, ax, Xmesh, Ymesh)

    logit,_ = encoder(torch.tensor(np.c_[Xmesh.reshape(-1, 1), Ymesh.reshape(-1,1)]).float().to('cuda'))
    pr = torch.nn.Softmax()(logit)[:, 1]
    pr = pr.data.cpu().numpy().reshape(100, 100)
    ax.contourf(Xmesh, Ymesh, pr, cmap='RdBu', alpha=0.3)
    ax.set_xlim([-2,2])
    ax.set_xlabel('n={}'.format(n))

def draw_eigen(ev_per_dim, cos_per_dim, ax, n_hidden_list):
    step = n_hidden_list
    #color=['salmon', 'green','royalblue']
    for i, data in enumerate(ev_per_dim):
        ax[0].plot(step, data, linestyle='-.')

    for i, data in enumerate(cos_per_dim):
        ax[1].plot(step, data, linestyle='-.')

def plot_countour(cov, ax, Xmesh, Ymesh, mean=(0,0), levels=[0.2,0.4,0.6,0.8,1.0]):
    # define grid.
    zi = np.array([pdf_multivariate_gauss(np.array([[x,y]]).T, np.array([[mean[0],mean[1]]]).T, cov) for x,y in zip(Xmesh.flatten(), Ymesh.flatten())])
    Zmesh = zi.reshape(100,100)

    CS = ax.contour(Xmesh, Ymesh, Zmesh, len(levels), linewidths=2., colors='grey', levels=levels)

def pdf_multivariate_gauss(x, mu, cov):
    part1 = 1 / ( ((2* np.pi)**(len(mu)/2)) * (np.linalg.det(cov)**(1/2)) )
    part2 = (-1/2) * ((x-mu).T.dot(np.linalg.inv(cov))).dot((x-mu))
    return float(part1 * np.exp(part2))

def confidence_ellipse(cov, mean, ax, n_std=3.0, facecolor='none', **kwargs):
    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensionl dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0),
        width=ell_radius_x * 2,
        height=ell_radius_y * 2,
        facecolor=facecolor,
        linewidth=2.,
        **kwargs)

    # Calculating the stdandard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = mean[0]

    # calculating the stdandard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = mean[1]

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)

def show_domain(X, y, ax, n_samples):
    ax.scatter(X[ :n_samples][y[ :n_samples]==0, 0], X[ :n_samples][y[ :n_samples]==0, 1], color='r', s=3)
    ax.scatter(X[ :n_samples][y[ :n_samples]==1, 0], X[ :n_samples][y[ :n_samples]==1, 1], color='b', s=3)

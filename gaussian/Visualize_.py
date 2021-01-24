import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
from gaussian.Data import mu, cov


def draw_decision_countour(encoder, ax, X, Y, n_samples, n):
    xi = np.linspace(-2., 2., 100)
    yi = np.linspace(-2., 2., 100)
    Xmesh, Ymesh = np.meshgrid(xi, yi)

    show_domain(X, Y, ax, n_samples)
    plot_gaussian(ax, Xmesh, Ymesh, mu, cov)
    plot_gaussian(ax, Xmesh, Ymesh, -mu, cov)

    logit = encoder(torch.tensor(np.c_[Xmesh.reshape(-1, 1), Ymesh.reshape(-1,1)]).float().to('cuda'))
    #pr = torch.nn.Softmax()(logit)[:, 1]
    pr = torch.nn.Sigmoid()(logit)
    pr = pr.data.cpu().numpy().reshape(100, 100)
    ax.contourf(Xmesh, Ymesh, pr, cmap='RdBu', alpha=0.3)
    ax.set_xlim([-2,2])
    ax.set_xlabel('n={}'.format(n))

def plot_gaussian(ax, Xmesh, Ymesh, mu, cov, levels=[0.2,0.4,0.6,0.8,1.0]):
    # define grid.
    zi = np.array([pdf_multivariate_gauss(np.array([[x,y]]).T, np.expand_dims(mu, 0).T, cov) \
                   for x,y in zip(Xmesh.flatten(), Ymesh.flatten())])
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

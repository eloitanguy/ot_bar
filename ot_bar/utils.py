import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse


def TN(x):
    """
    Returns a numpy version of the array or list of arrays given as input

    Args:
        x: torch tensor or list thereof
    """
    if isinstance(x, list) or isinstance(x, tuple):
        return [TN(o) for o in x]

    if torch.is_tensor(x):
        return x.detach().cpu().numpy()

    if isinstance(x, np.ndarray):
        return x

    raise TypeError('Expected a numpy array or a torch tensor')


def TT(x):
    """
    Returns a torch version (cuda if possible and dtype = double)
    of the array or list of arrays given as input

    Args:
        x: numpy tensor or list thereof
    """
    if isinstance(x, list) or isinstance(x, tuple):
        return [TT(o) for o in x]

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if isinstance(x, np.ndarray):
        return torch.tensor(x, dtype=torch.double, device=device)

    if torch.is_tensor(x):
        if device not in str(x.device):  # check if on the right device
            x = x.to(device)
        if x.dtype != torch.double:
            x = x.double()
        return x

    raise TypeError('Expected a numpy array or a torch tensor')


def imageToGrid(image, constant_threshold=True):
    """
    Convert an image to an array (n_points,2) of points via thresholding
    """

    def condition(p):
        if constant_threshold:
            return p < 122.5
        return p > np.max(image) * 0.5

    x, y = image.shape

    points = []

    for i in range(x):
        for j in range(y):
            if condition(image[i, j]):
                points.append([1 - 1. * i / x, 1. * j / y])

    return np.array(points)


def plot_runs(runs, x=None, ax=None, curve_labels=None, title='', x_label='',
              x_scale_log=False, y_scale_log=False):
    r"""
    Plots runs, a numpy array of size (n_curve_params, n_x_params, n_runs),
    corresponding to experiments results with different samples for each
    parameter value for the total n_curve_params * n_x_params parameter values.
    For each parameter in n_curve_params, this plots the median and 30% / 70%
    quantiles as a function of the x parameter. The array x of size n_x_params
    corresponds to the x-axis values.
    """
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(5, 5))
    n_y, n_x, n_runs = runs.shape
    if curve_labels is None:
        curve_labels = [None] * n_y
    if x is None:
        x = np.arange(n_x)
    for run, label in zip(runs, curve_labels):
        ax.plot(x,
                np.median(run, axis=0),
                label=label)
        ax.fill_between(x,
                        np.quantile(run, .3, axis=0),
                        np.quantile(run, .7, axis=0),
                        alpha=.3)

    ax.set_title(title)
    ax.set_xlabel(x_label)
    if x_scale_log:
        plt.xscale('log')
    if y_scale_log:
        plt.yscale('log')
    plt.legend()
    plt.tight_layout()


def get_random_GMM(K, d, seed=0, min_cov_eig=1, cov_scale=1e-2):
    rng = np.random.RandomState(seed=seed)
    means = rng.randn(K, d)
    P = rng.randn(K, d, d) * cov_scale
    # C[k] = P[k] @ P[k]^T + min_cov_eig * I
    covariances = np.einsum('kab,kcb->kac', P, P)
    covariances += min_cov_eig * np.array([np.eye(d) for _ in range(K)])
    weights = rng.random(K)
    weights /= np.sum(weights)
    return means, covariances, weights


def draw_cov(mu, C, color=None, label=None, nstd=1, alpha=0.5):
    def eigsorted(cov):
        if torch.is_tensor(cov):
            cov = cov.detach().numpy()
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1].copy()
        return vals[order], vecs[:, order]

    vals, vecs = eigsorted(C)
    theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
    w, h = 2 * nstd * np.sqrt(vals)
    ell = Ellipse(
        xy=(mu[0], mu[1]),
        width=w,
        height=h,
        alpha=alpha,
        angle=theta,
        facecolor=color,
        edgecolor=color,
        label=label,
        fill=True,
    )
    plt.gca().add_artist(ell)


def draw_gmm(ms, Cs, ws, color=None, nstd=0.5, alpha=1, label=None):
    for k in range(ms.shape[0]):
        draw_cov(ms[k], Cs[k], color, label if k == 0 else None,
                 nstd, alpha * ws[k])

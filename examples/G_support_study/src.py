# %%
import numpy as np
from tqdm import trange
from ot_bar.solvers import solve_OT_barycenter_fixed_point
import matplotlib.pyplot as plt
from ot_bar.utils import sample_simplex
import ot  # type: ignore
import os


def c(x, y):
    return ot.dist(x, y)


def B(y):
    return np.mean(y, axis=0)  # isobarycentre is the mean of the points


def sample_setup(n_choices, K_choices, nk_choices, d_choices, seed=0,
                 point_clouds=False):
    """
    Sample a setup for the barycenter problem.
    """
    np.random.seed(seed)
    n = np.random.choice(n_choices)
    K = np.random.choice(K_choices)
    if point_clouds:
        nk_list = [n] * K
    else:
        nk_list = np.random.choice(nk_choices, K, replace=True)
    d = np.random.choice(d_choices)

    X_init = np.random.randn(n, d)
    Y_list = [np.random.randn(nk, d) for nk in nk_list]

    if point_clouds:
        a_init = ot.unif(n)
        b_list = [ot.unif(nk) for nk in nk_list]
    else:
        a_init = sample_simplex(n)
        b_list = [sample_simplex(nk) for nk in nk_list]

    return X_init, a_init, Y_list, b_list


def support_ratios(X_init, a_init, Y_list, b_list, max_its, stop_threshold):
    K = len(Y_list)
    cost_list = [c] * K
    _, a, log_dict = solve_OT_barycenter_fixed_point(
        X_init, Y_list, b_list, cost_list, B, a=a_init, max_its=max_its,
        stop_threshold=stop_threshold, method='true_fixed_point',
        clean_measure=True, log=True)
    N = a.shape[0]
    # number of iterations (not counting the initial point)
    T = len(log_dict['X_list']) - 1
    nk_list = [Y.shape[0] for Y in Y_list]
    N_init = X_init.shape[0]
    ratio_theory = N / (N_init + T * sum(nk_list) - T * K)
    ratio_mm = N / (sum(nk_list) - K + 1)
    return ratio_theory, ratio_mm, T, N / N_init


# %% Free ratio experiment
n_samples = 500
n_choices = np.arange(10, 100)
K_choices = np.arange(2, 10)
nk_choices = np.arange(10, 100)
d_choices = np.arange(1, 20)


if not os.path.exists('free_ratio_results.npy'):
    # dim 0: samples, dim 1: [theory, MM, T, init]
    free_ratio_results = np.zeros((n_samples, 4))
    for i in trange(n_samples, desc='Free ratio experiment'):
        X_init, a_init, Y_list, b_list = sample_setup(
            n_choices, K_choices, nk_choices, d_choices, seed=i)
        ratio_theory, ratio_mm, T, ratio_init = support_ratios(
            X_init, a_init, Y_list, b_list, max_its=10, stop_threshold=1e-2)
        free_ratio_results[i, 0] = ratio_theory
        free_ratio_results[i, 1] = ratio_mm
        free_ratio_results[i, 2] = T
        free_ratio_results[i, 3] = ratio_init
    np.save('free_ratio_results.npy', free_ratio_results)
else:
    free_ratio_results = np.load('free_ratio_results.npy')


# %% plot results with violin plot
plt.figure(figsize=(12, 4))
violinplot_kwargs = {
    'showmeans': False,
    'showmedians': True,
}

# Ratio vs N_init
plt.subplot(1, 3, 1)
plt.violinplot(free_ratio_results[:, 3], **violinplot_kwargs)
plt.xticks([1], ['$N\ /\ N_0$'])
plt.ylabel('Support Ratio')
plt.title('Ratio with $N_0$')
plt.grid(True)

# Theory ratio
plt.subplot(1, 3, 2)
plt.violinplot(free_ratio_results[:, 0], **violinplot_kwargs)
plt.xticks([1], ['$N\ /\ \left(N_0 + T\sum_k n_k - TK\\right)$'])
plt.ylabel('Support Ratio')
plt.title('Ratio with Theory')
plt.grid(True)

# MM ratio
plt.subplot(1, 3, 3)
plt.violinplot(free_ratio_results[:, 1], **violinplot_kwargs)
plt.xticks([1], ['$N\ /\ \left(\sum_k n_k - K + 1\\right)$'])
plt.title('Ratio with MM Theory')
plt.grid(True)

plt.tight_layout()
plt.savefig('support_ratios_violin_plot.pdf', bbox_inches='tight')
plt.show()

# %% plot number of iterations histogram
histogram_kwargs = {
    'edgecolor': 'black',
    'alpha': 0.7,
}
plt.figure(figsize=(4, 4))
plt.bar(*np.unique(free_ratio_results[:, 2], return_counts=True),
        **histogram_kwargs)
plt.xlabel('Number of Iterations')
plt.ylabel('Frequency')
plt.grid(True)
plt.tight_layout()
plt.savefig('support_ratios_iterations_histogram.pdf', bbox_inches='tight')
plt.show()

# %% point cloud ratio experiment
n_samples = 500
n_choices = np.arange(10, 100)
K_choices = np.arange(2, 10)
nk_choices = np.arange(10, 100)
d_choices = np.arange(1, 20)


if not os.path.exists('point_cloud_ratio_results.npy'):
    # dim 0: samples, dim 1: [theory, MM, T, init]
    point_cloud_ratio_results = np.zeros((n_samples, 4))
    for i in trange(n_samples, desc='Free ratio experiment'):
        X_init, a_init, Y_list, b_list = sample_setup(
            n_choices, K_choices, nk_choices, d_choices, seed=i,
            point_clouds=True)
        ratio_theory, ratio_mm, T, ratio_init = support_ratios(
            X_init, a_init, Y_list, b_list, max_its=10, stop_threshold=1e-2)
        point_cloud_ratio_results[i, 0] = ratio_theory
        point_cloud_ratio_results[i, 1] = ratio_mm
        point_cloud_ratio_results[i, 2] = T
        point_cloud_ratio_results[i, 3] = ratio_init
    np.save('point_cloud_ratio_results.npy', point_cloud_ratio_results)
else:
    point_cloud_ratio_results = np.load('point_cloud_ratio_results.npy')

# %% plot results with violin plot
plt.figure(figsize=(12, 4))

# Ratio vs N_init
plt.subplot(1, 3, 1)
plt.violinplot(point_cloud_ratio_results[:, 3], **violinplot_kwargs)
plt.xticks([1], ['$N\ /\ N_0$'])
plt.ylabel('Support Ratio')
plt.title('Ratio with $N_0$')
plt.grid(True)

# Theory ratio
plt.subplot(1, 3, 2)
plt.violinplot(point_cloud_ratio_results[:, 0], **violinplot_kwargs)
plt.xticks([1], ['$N\ /\ \left(N_0 + T\sum_k n_k - TK\\right)$'])
plt.ylabel('Support Ratio')
plt.title('Ratio with Theory')
plt.grid(True)

# MM ratio
plt.subplot(1, 3, 3)
plt.violinplot(point_cloud_ratio_results[:, 1], **violinplot_kwargs)
plt.xticks([1], ['$N\ /\ \left(\sum_k n_k - K + 1\\right)$'])
plt.title('Ratio with MM Theory')
plt.grid(True)

plt.tight_layout()
plt.savefig('support_ratios_violin_plot_point_clouds.pdf', bbox_inches='tight')
plt.show()

# %% plot number of iterations histogram
plt.figure(figsize=(4, 4))
plt.bar(*np.unique(point_cloud_ratio_results[:, 2], return_counts=True),
        **histogram_kwargs)
plt.xlabel('Number of Iterations')
plt.ylabel('Frequency')
plt.grid(True)
plt.tight_layout()
plt.savefig('support_ratios_iterations_histogram_point_clouds.pdf',
            bbox_inches='tight')
plt.show()

# %%

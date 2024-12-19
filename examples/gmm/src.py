# %%
import numpy as np
import ot
from ot_bar.solvers import solve_gmm_barycentre_multi_marginal, solve_gmm_barycenter_fixed_point
from ot_bar.utils import draw_gmm, get_random_gmm, draw_gmm_contour
import matplotlib.pyplot as plt
from ot.gmm import gmm_ot_loss
from time import time
from sklearn.mixture import GaussianMixture


K = 3
d = 2

m_list = [5, 6, 7]
offsets = [np.array([-3, 0]), np.array([2, 0]), np.array([0, 4])]
means_list = []
covs_list = []
b_list = []

for k in range(K):
    means, covs, b = get_random_gmm(m_list[k], d, seed=k, min_cov_eig=.25,
                                    cov_scale=.5)
    means = means / 2 + offsets[k][None, :]
    means_list.append(means)
    covs_list.append(covs)
    b_list.append(b)

weights = ot.unif(K)

# %% Compare multi-marginal and fixed point barycenters
fig, axs = plt.subplots(1, 2, figsize=(12, 6))

# First subplot for multi-marginal barycenter
t0 = time()
means_bar, covs_bar, a = solve_gmm_barycentre_multi_marginal(
    means_list, covs_list, b_list, weights)
dt = time() - t0
axis = [-4, 4, -2, 6]
axs[0].set_title(f'Multi-Marginal Barycenter, time: {dt:.3f}s',
                 fontsize=16)
for k in range(K):
    draw_gmm(means_list[k], covs_list[k], b_list[k], color='C0', ax=axs[0])
draw_gmm(means_bar, covs_bar, a, color='C1', ax=axs[0])
axs[0].axis(axis)
axs[0].axis('off')

# Second subplot for fixed point barycenter
n = 6
fixed_its = 3
init_means, init_covs, _ = get_random_gmm(n, d, seed=0)

t0 = time()
means_bar, covs_bar, log = solve_gmm_barycenter_fixed_point(
    init_means, init_covs,
    means_list, covs_list, b_list, weights, max_its=fixed_its, log=True)
dt = time() - t0

axs[1].set_title(f'Fixed Point Barycenter, time: {dt:.3f}s',
                 fontsize=16)
for k in range(K):
    draw_gmm(means_list[k], covs_list[k], b_list[k], color='C0', ax=axs[1])
draw_gmm(means_bar, covs_bar, ot.unif(n), color='C1', ax=axs[1])
axs[1].axis(axis)
axs[1].axis('off')

plt.tight_layout()
plt.savefig('gmm_barycenters_comparison.pdf', bbox_inches='tight')
plt.show()

# %% energy V per iteration
V_list = []
for it in range(fixed_its):
    V = 0
    for k in range(K):
        V += gmm_ot_loss(log['means_its'][it], means_list[k],
                         log['covs_its'][it], covs_list[k],
                         ot.unif(n), b_list[k])
    V_list.append(V)

plt.plot(V_list)
plt.xlabel('iteration')
plt.ylabel('V')
plt.savefig('gmm_fixed_point_V.pdf')

# %% GMM barycentre grid: data
im1 = 1 - plt.imread('../data/redcross.png')[:, :, 2]
im2 = 1 - plt.imread('../data/duck.png')[:, :, 2]
im3 = 1 - plt.imread('../data/fire.png')[:, :, 2]
im4 = 1 - plt.imread('../data/heart.png')[:, :, 2]
images = [im1, im2, im3, im4]

n_components = 15  # number of components for EM
n = 15
a = ot.unif(n)
fixed_its = 3
means_list = []
covs_list = []
b_list = []

for im in images:  # fit GMM with EM
    ind = np.nonzero(im[::-1, :])
    X = np.zeros((ind[0].shape[0], 2))
    X[:, 0] = ind[1]
    X[:, 1] = ind[0]
    gmm = GaussianMixture(n_components=n_components, covariance_type='full',
                          random_state=0).fit(X)
    means_list.append(gmm.means_)
    covs_list.append(gmm.covariances_)
    b_list.append(gmm.weights_)

K = len(images)
plt.figure(figsize=(12, 3))

for k in range(K):
    axes = plt.subplot(1, K, k + 1)
    m, C, w = means_list[k], covs_list[k], b_list[k]
    draw_gmm_contour(m, C, w, n=128, ax=-5, bx=132, ay=-5, by=132)
    plt.axis('off')
plt.tight_layout()
plt.savefig('gmms.pdf', bbox_inches='tight')

# %% GMM barycentre grid: computation
n_grid = 7
# four corners that will be interpolated by bilinear interpolation
v1 = np.array((1, 0, 0, 0))
v2 = np.array((0, 1, 0, 0))
v3 = np.array((0, 0, 1, 0))
v4 = np.array((0, 0, 0, 1))

# barycentre results
bars_dict = {}
Kn = k**4

print('start barycenter computation')
t0 = time()
for k in range(n_grid):
    for j in range(n_grid):
        if k == 0 and j == 0:
            bars_dict[(k, j)] = means_list[0], covs_list[0], b_list[0]

        elif k == 0 and j == n_grid - 1:
            bars_dict[(k, j)] = means_list[2], covs_list[2], b_list[2]

        elif k == n_grid - 1 and j == 0:
            bars_dict[(k, j)] = means_list[1], covs_list[1], b_list[1]

        elif k == n_grid - 1 and j == (n_grid - 1):
            bars_dict[(k, j)] = means_list[3], covs_list[3], b_list[3]

        else:
            tx = float(k) / (n_grid - 1)
            ty = float(j) / (n_grid - 1)

            # weights are constructed by bilinear interpolation
            tmp1 = (1 - tx) * v1 + tx * v2
            tmp2 = (1 - tx) * v3 + tx * v4
            weights = (1 - ty) * tmp1 + ty * tmp2
            init_means, init_covs, _ = get_random_gmm(n, d, seed=0)
            m, C = solve_gmm_barycenter_fixed_point(
                init_means, init_covs,
                means_list, covs_list, b_list, weights, max_its=fixed_its)
            bars_dict[(k, j)] = m, C, a

print(f'end barycenter computation in: {time() - t0:.5f}s')

plt.figure(figsize=(20, 20))

for k in range(n_grid):
    for j in range(n_grid):
        axes = plt.subplot(n_grid, n_grid, k * n_grid + j + 1)
        m, C, w = bars_dict[(k, j)]
        draw_gmm_contour(m, C, w, n=128, ax=-5, bx=132, ay=-5, by=132)
        plt.axis('off')
plt.tight_layout()
plt.savefig('gmm_barycenter_interpolation.pdf', bbox_inches='tight')

# %%

# %%
import numpy as np
import ot
from ot_bar.solvers import solve_gmm_barycentre_multi_marginal
from ot_bar.utils import draw_gmm
import matplotlib.pyplot as plt


K = 3
d = 2

m0 = np.array([
    [-1, 0],
    [1, 0],
    [0, 1]
])
w0 = ot.unif(3)
C0 = np.array([
    [[1, .1],
     [.1, 1]],
    [[1.2, -.1],
     [-.1, 1]],
    [[.8, 0],
     [0, 1.1]]
])

m1 = np.array([
    [5, 0],
    [5.5, 1.5],
])
w1 = ot.unif(2)
C1 = np.array([
    [[1, 0],
     [0, 1]],
    [[2, .2],
     [.2, 1]],
])

m2 = np.array([
    [1.75, 5],
    [2, 6],
    [3, 4.5],
    [2.5, 5.5]
])
w2 = ot.unif(4)
C2 = np.array([
    [[1, 0],
     [0, 1]],
    [[2, .2],
     [.2, 1]],
    [[1, -.2],
     [-.2, 1.1]],
    [[1.5, -.2],
     [-.2, .8]]
])
means_list = [m0, m1, m2]
covs_list = [C0, C1, C2]
w_list = [w0, w1, w2]
weights = ot.unif(K)

# %%
means_bar, covs_bar, a = solve_gmm_barycentre_multi_marginal(
    means_list, covs_list, w_list, weights)

axis = [-2, 7, -1, 7]
plt.figure(1, (6, 6))
plt.clf()
for k in range(K):
    draw_gmm(means_list[k], covs_list[k], w_list[k], color='C0')
draw_gmm(means_bar, covs_bar, a, color='C1', label='barycenter')
plt.axis(axis)
plt.legend()
plt.show()
plt.savefig('gmm_barycenter.pdf')

# %%

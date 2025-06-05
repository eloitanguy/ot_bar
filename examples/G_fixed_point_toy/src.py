# %%
import numpy as np
from ot.backend import get_backend  # type: ignore
import ot  # type: ignore
from tqdm import trange
from time import time
from ot_bar.solvers import NorthWestMMGluing, solve_OT_barycenter_fixed_point, solve_w2_barycentre_multi_marginal
import matplotlib.pyplot as plt


c_list = ['#7ED321', '#4A90E2', '#9013FE']
c_bar = '#D0021B'


def test_gluing_validity(gamma, J, w, pi_list):
    """
    Test the validity of the North-West gluing.
    """
    nx = get_backend(gamma)
    K = len(pi_list)
    n = pi_list[0].shape[0]
    nk_list = [pi.shape[1] for pi in pi_list]

    # Check first marginal
    a = nx.sum(gamma, axis=tuple(range(1, K + 1)))
    assert nx.allclose(a, nx.sum(pi_list[0], axis=1))

    # Check other marginals
    for k in range(K):
        b_k = nx.sum(gamma, axis=tuple(i for i in range(K + 1) if i != k + 1))
        assert nx.allclose(b_k, nx.sum(pi_list[k], axis=0))

    # Check bi-marginals
    for k in range(K):
        gamma_0k = nx.sum(gamma,
                          axis=tuple(i for i in range(1, K + 1) if i != k + 1))
        assert nx.allclose(gamma_0k, pi_list[k])

    # Check that N <= n + sum_k n_k - K
    N = J.shape[0]
    n_k_sum = sum(nk_list)
    assert N <= n + n_k_sum - K, f"N={N}, n={n}, sum(n_k)={n_k_sum}, K={K}"

    # Check that each w is on the simplex
    w_sum = nx.sum(w)
    assert nx.allclose(w_sum, 1), f"Sum of weights w is not 1: {w_sum}"

    # Check that gamma_1...K and (J, w) are consistent
    rho = nx.zeros(nk_list, type_as=gamma)
    for i in range(N):
        jj = J[i]
        rho[tuple(jj)] += w[i]

    gamma_1toK = nx.sum(gamma, axis=0)
    assert nx.allclose(rho, gamma_1toK), "rho and gamma_1...K are not consistent"


# %%
np.random.seed(0)

for _ in trange(100):
    K = np.random.randint(2, 5)  # number of marginals
    n = np.random.randint(5, 30)  # number of points in the first marginal
    # number of points in other marginals
    nk_list = [np.random.randint(5, 30) for _ in range(K)]
    a = np.random.rand(n)
    a = a / np.sum(a)
    b_list = [np.random.rand(nk) for nk in nk_list]
    b_list = [b / np.sum(b) for b in b_list]
    M_list = [np.random.rand(n, nk) for nk in nk_list]
    pi_list = [ot.emd(a, b, M) for b, M in zip(b_list, M_list)]
    J, w, log_dict = NorthWestMMGluing(pi_list, log=True)
    # Test the validity of the gluing
    gamma = log_dict['gamma']
    test_gluing_validity(gamma, J, w, pi_list)


# %% Larger scale without logging
np.random.seed(0)
t_init = time()
K = 5  # number of marginals
n = 1000  # number of points in the first marginal
nk_list = [n] * K  # number of points in other marginals
a = np.random.rand(n)
a = a / np.sum(a)
b_list = [np.random.rand(nk) for nk in nk_list]
b_list = [b / np.sum(b) for b in b_list]
M_list = [np.random.rand(n, nk) for nk in nk_list]
pi_list = [ot.emd(a, b, M) for b, M in zip(b_list, M_list)]
t0 = time()
print(f"Setup with K={K} and n={n} took {t0 - t_init:.2f} seconds")
J, w = NorthWestMMGluing(pi_list)
t1 = time()
print(f"Time taken for North-West gluing with K={K} and n={n}:\n"
      f"\t{t1 - t0:.2f} seconds, support size: {J.shape[0]}\n"
      f"\twith n + sum_k n_k - K = {n + sum(nk_list) - K}")

# %% Pathological case of independent couplings
np.random.seed(0)
K = 3  # number of marginals
n = 20  # number of points in the first marginal
nk_list = [n] * K  # number of points in other marginals
a = np.random.rand(n)
a = a / np.sum(a)
b_list = [np.random.rand(nk) for nk in nk_list]
b_list = [b / np.sum(b) for b in b_list]
pi_list = [a[:, None] * b[None, :] for b in b_list]  # independent couplings
J, w, log_dict = NorthWestMMGluing(pi_list, log=True)
# Test the validity of the gluing
gamma = log_dict['gamma']
test_gluing_validity(gamma, J, w, pi_list)

# %% Toy example with 2 marginals
np.random.seed(0)
K = 2
n1 = 10
t = np.linspace(0, 2 * np.pi, n1)
offset1 = 3 * np.array([-1., -1.])[None, :]
Y1 = offset1 + np.array([np.cos(t), np.sin(t)]).T  # Circle 1
n2 = 15
t = 3 * np.linspace(0, 2 * np.pi, n2)
offset2 = np.array([1., 1.])[None, :]
s = 7 * np.pi / 13
R = np.array([[np.cos(s), -np.sin(s)],
              [np.sin(s), np.cos(s)]])
Y2 = offset2 + np.array([np.cos(t), np.sin(t)]).T @ R  # Circle 2
Y_list = [Y1, Y2]
b_list = [ot.unif(n1), ot.unif(n2)]


def c(x, y):
    return ot.dist(x, y)


cost_list = [c, c]


def B(y):
    return np.mean(y, axis=0)  # isobarycentre is the mean of the points


n = 10  # initial number of points in the barycentre
a = ot.unif(n)
X_init = np.random.rand(n, 2)  # Initial barycentre points
X, a, log_dict = solve_OT_barycenter_fixed_point(
    X_init, Y_list, b_list, cost_list, B, a=a, max_its=10, pbar=True, log=True,
    method='true_fixed_point', clean_measure=True, stop_threshold=1e-3)

# %% Compute the true barycentre
X_true, a_true = solve_w2_barycentre_multi_marginal(
    Y_list, b_list, ot.unif(K), clean_measure=True)

# %% Plot the iterations and true MM barycentre
n_its = len(log_dict['X_list'])
size = 4
fig, axes = plt.subplots(1, n_its, figsize=((n_its + 1) * size, 5))

X_list = log_dict['X_list']
a_list = log_dict['a_list']
base_marker_size = 800

# compute barycentre costs
bar_cost_list = []
for t in range(n_its):
    bar_cost = 0
    for k in range(K):
        bar_cost += ot.emd2(a_list[t], b_list[k],
                            cost_list[k](X_list[t], Y_list[k]))
    bar_cost_list.append(bar_cost / K)

# plot iterations
for t in range(n_its):
    ax = axes[t]
    for k in range(K):
        ax.scatter(Y_list[k][:, 0], Y_list[k][:, 1],
                   s=base_marker_size * b_list[k],
                   color=c_list[k], label=f'Measure {k + 1}')
    ax.scatter(X_list[t][:, 0], X_list[t][:, 1],
               s=base_marker_size * a_list[t],
               color=c_bar, alpha=0.5, label='Barycentre')
    ax.axis('equal')
    ax.axis('off')
    ax.set_title(
        f"Iteration {t},\nSupport: {X_list[t].shape[0]}"
        f"\nCost:{bar_cost_list[t]:.4f}", fontsize=18)

# true barycentre cost
bar_cost_true = 0
for k in range(K):
    bar_cost_true += ot.emd2(a_true, b_list[k],
                             cost_list[k](X_true, Y_list[k]))
bar_cost_true /= K

# plot true barycentre
axes[-1].scatter(X_true[:, 0], X_true[:, 1],
                 s=base_marker_size * a_true,
                 color=c_list[-1], alpha=0.5, label='True Barycentre')
axes[-1].set_title(
    f"True Barycentre\nSupport: {X_true.shape[0]}"
    f"\nCost: {bar_cost_true:.4f}", fontsize=18)

# Add a common legend to the figure
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', ncol=K + 1, fontsize=18,
           bbox_to_anchor=(0.5, -0.1))

plt.tight_layout()
plt.savefig('true_fp_toy_iterations.pdf', format='pdf', bbox_inches='tight')
plt.show()

# %%

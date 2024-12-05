# %% load and generate data. Adapted from
# https://pythonot.github.io/auto_examples/barycenters/plot_free_support_barycenter.html
import numpy as np
import matplotlib.pylab as pl
import matplotlib.pyplot as plt
import ot
from ot_bar.utils import TT, TN
from ot_bar.solvers import solve_OT_barycenter_fixed_point, StoppingCriterionReached
import torch
from torch.optim import Adam
from matplotlib import cm
from tqdm import tqdm

np.random.seed(42)
torch.manual_seed(42)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

K = 2
d = 2
n = 400

I1 = pl.imread("../data/redcross.png").astype(np.float64)[::4, ::4, 2]
I2 = pl.imread("../data/duck.png").astype(np.float64)[::4, ::4, 2]

sz = I2.shape[0]
XX, YY = np.meshgrid(np.arange(sz), np.arange(sz))

Y1 = np.stack((XX[I1 == 0], YY[I1 == 0]), 1) * 1.0
Y2 = np.stack((XX[I2 == 0] + 80, -YY[I2 == 0] + 32), 1) * 1.0
Y1 = Y1 + np.random.randn(*Y1.shape) * .3
Y2 = Y2 + np.random.randn(*Y2.shape) * .3

Y_list = TT([Y1, Y2])
n1, n2 = Y1.shape[0], Y2.shape[0]
b1, b2 = ot.unif(n1), ot.unif(n2)
b_list = TT([b1, b2])
a = TT(ot.unif(n))

plt.figure(1, (12, 4))
plt.scatter(*TN(Y1.T), alpha=0.5)
plt.scatter(*TN(Y2.T), alpha=0.5)
plt.title("Distributions")

# %% Define costs and visualise landscapes
p_list = [1.5, 2, 3]  # p-norms
q_list = [1.5, 2, 3]  # exponents to the p-norms


def p_norm_q_cost_matrix(u, v, p, q):
    return torch.sum(torch.abs(u[:, None, :] - v[None, :, :])**p, axis=-1)**(q / p)


def C(x, y, p, q):
    """
    Computes the barycenter cost for candidate points x (n, d) and
    measure supports y: List(n, d_k).
    """
    n = x.shape[0]
    K = len(y)
    out = torch.zeros(n, device=device)
    for k in range(K):
        out += (1 / K) * torch.sum(torch.abs(x - y[k])**p, axis=1)**(q / p)
    return out


n_vis = 100
u = torch.linspace(40, 80, n_vis, device=device)
v = torch.linspace(0, 30, n_vis, device=device)
uu, vv = torch.meshgrid(u, v)
x = torch.stack([uu.flatten(), vv.flatten()], dim=1)
y_idx = [n // 5, 2 * n // 5, 3 * n // 5, 4 * n // 5]
y = [(Y_list[k][y_idx[k]])[None, :] * torch.ones_like(x, device=device) for k in range(K)]
uu_np, vv_np = TN(uu), TN(vv)

fig = plt.figure(figsize=(len(p_list) * 3, len(q_list) * 3))

for p_idx, p in enumerate(p_list):
    for q_idx, q in enumerate(q_list):
        i = p_idx * len(q_list) + q_idx
        ax = fig.add_subplot(len(p_list), len(q_list), i + 1, projection='3d')
        M = C(x, y, p, q)  # shape (n_vis**2)
        M = TN(M.reshape(n_vis, n_vis))
        ax.plot_surface(uu_np, vv_np, M, cmap=cm.CMRmap, edgecolor='none')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        ax.set_title(f'C p={p}, q={q}', fontsize=16)

plt.savefig("B_energy_map.pdf", format="pdf", bbox_inches="tight")
plt.show(block=True)


# %% Define ground barycentre function B and test loss decrease
def B(y, p, q, its=250, lr=1, log=False, stop_threshold=1e-20):
    """
    Computes the barycenter images for candidate points x (n, d) and
    measure supports y: List(n, d_k).
    Output: (n, d) array
    """
    x = torch.randn(n, d, device=device, dtype=torch.double)
    x.requires_grad_(True)
    loss_list = [1e10]
    opt = Adam([x], lr=lr)
    exit_status = 'unknown'
    try:
        for _ in range(its):
            opt.zero_grad()
            loss = torch.sum(C(x, y, p, q))
            loss.backward()
            opt.step()
            loss_list.append(loss.item())
            if stop_threshold > loss_list[-2] - loss_list[-1] >= 0:
                exit_status = 'Local optimum'
                raise StoppingCriterionReached
        exit_status = 'Max iterations reached'
        raise StoppingCriterionReached
    except StoppingCriterionReached:
        if log:
            return x, {'loss_list': loss_list[1:], 'exit_status': exit_status}
        return x


np.random.seed(42)
torch.manual_seed(42)
fig = plt.figure(figsize=(len(p_list) * 3, len(q_list) * 3))
X_init = torch.rand(n, d, device=device, dtype=torch.double)

for p_idx, p in enumerate(p_list):
    for q_idx, q in enumerate(q_list):
        i = p_idx * len(q_list) + q_idx
        pi_list = [ot.emd(a, b_list[k], p_norm_q_cost_matrix(X_init, Y_list[k], p, q)) for k in range(K)]
        Y_perm = []
        for k in range(K):
            Y_perm.append(n * pi_list[k] @ Y_list[k])
        Bx, log = B(Y_perm, p, q, log=True)
        ax = fig.add_subplot(len(p_list), len(q_list), i + 1)
        ax.plot(log['loss_list'])
        ax.set_yscale('log')
        ax.set_title(f'B loss p={p}, q={q}', fontsize=12)

plt.savefig('B_losses.pdf')

# %% apply fixed-point algorithm
np.random.seed(0)
torch.manual_seed(0)
fixed_point_its = 5
X_bar_dict = {}
X_bar_list_dict = {}
for p in tqdm(p_list):
    for q in q_list:
        cost_list = [lambda x, y: p_norm_q_cost_matrix(x, y, p, q)] * K
        X_bar, X_bar_list = solve_OT_barycenter_fixed_point(
            X_init, Y_list, b_list, cost_list, lambda y: B(y, p, q), max_its=fixed_point_its, log=True)
        X_bar_dict[(p, q)] = X_bar
        X_bar_list_dict[(p, q)] = X_bar_list

# %% plot barycentres
fig = plt.figure(figsize=(len(p_list) * 3, len(q_list) * 3))
for p_idx, p in enumerate(p_list):
    for q_idx, q in enumerate(q_list):
        i = p_idx * len(q_list) + q_idx
        ax = fig.add_subplot(len(p_list), len(q_list), i + 1)
        X_bar = X_bar_dict[(p, q)]
        plt.scatter(*TN(X_bar.T), alpha=0.5)
        ax.set_title(f'Barycentre p={p}, q={q}', fontsize=12)
        ax.axis('equal')
        ax.axis('off')
plt.savefig('p_norm_q_barycentres.pdf')

# %% plot energy evolution
fig = plt.figure(figsize=(len(p_list) * 3, len(q_list) * 3))
for p_idx, p in enumerate(p_list):
    for q_idx, q in enumerate(q_list):
        i = p_idx * len(q_list) + q_idx
        ax = fig.add_subplot(len(p_list), len(q_list), i + 1)
        V_list = []
        for X_bar in X_bar_list_dict[(p, q)]:
            V = 0
            for k in range(K):
                M = p_norm_q_cost_matrix(X_bar, Y_list[k], p, q)
                V += (1 / K) * ot.emd2(a, b_list[k], M)
            V_list.append(V.item())
        ax.plot(V_list)
        ax.set_yscale('log')
        ax.set_title(f'V p={p}, q={q}', fontsize=12)
plt.savefig('p_norm_q_barycentres_V.pdf')

# %%

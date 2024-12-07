# %%
import numpy as np
import torch
from ot_bar.solvers import solve_NLGWB_GD, solve_OT_barycenter_fixed_point, solve_barycentre_multi_marginal
from ot_bar.utils import TT, TN
from time import time
import matplotlib.pyplot as plt
import ot  # type: ignore


np.random.seed(42)
torch.manual_seed(42)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
K = 3
d = 2
n = 50
m_list = [50, 51, 52]
b_list = [TT(ot.unif(m)) for m in m_list]
weights = TT(ot.unif(K))

Y_list = []
offsets = TT([np.array([-5, 0]), np.array([5, 0]), np.array([0, 5])])
for m, dy in zip(m_list, offsets):
    Y = torch.randn(m, d, device=device, dtype=torch.float64)
    Y = Y + dy[None, :]
    Y_list.append(Y)

(p, q) = 2, 2


def cost_matrix(u, v):
    return torch.sum(torch.abs(u[:, None, :] - v[None, :, :])**p, axis=-1)**(q / p)


def C(x, y, p, q):
    """
    Computes the ground barycenter cost for candidate points x (n, d) and
    measure supports y: List(n, d_k).
    """
    n = x.shape[0]
    K = len(y)
    out = torch.zeros(n, device=device)
    for k in range(K):
        out += (1 / K) * torch.sum(torch.abs(x - y[k])**p, axis=1)**(q / p)
    return out


for y in Y_list:
    plt.scatter(*TN(y.T), alpha=0.5)
plt.title("Distributions")
plt.show()

# %%
t = time()
X, a = solve_barycentre_multi_marginal(TN(Y_list), TN(b_list), TN(weights))
print(f'Solved MMbar in: {time() - t:.5f}s')
for y in Y_list:
    plt.scatter(*TN(y.T), alpha=0.5)
plt.scatter(*TN(X.T), alpha=0.5)
plt.title("MMOT bar")
plt.show()

# %%

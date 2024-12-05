# %%
import numpy as np
import torch
from ot_bar.solvers import solve_NLGWB_GD, solve_OT_barycenter_fixed_point, StoppingCriterionReached
from ot_bar.utils import TT, TN
from time import time
import matplotlib.pyplot as plt
import ot  # type: ignore
from torch.optim import Adam
import matplotlib.animation as animation
from matplotlib import cm
from itertools import product
import cvxpy as cvx


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


# %% Multi-marginal solver: define the cost tensor
def multi_marginal_cost_tensor(y, weights):
    """
    Computes the m_1 x ... x m_K tensor of costs for the multi-marginal problem
    for the L2 cost.
    """
    m_list = [y[k].shape[0] for k in range(K)]
    M = np.zeros(m_list)
    for indices in product(*[range(m) for m in m_list]):
        # indices is (j1, ..., jK)
        # y_slice is a K x d matrix Y1[j1] ... YK[jK]
        y_slice = np.stack([y[k][indices[k]] for k in range(K)], axis=0)
        mean = weights @ y_slice  # (1, d)
        norms = np.sum((mean - y_slice)**2, axis=1)  # (K,)
        M[indices] = np.sum(weights * norms)
    return M


def solve_MMOT(b_list, M):
    """
    Solves the Multi-Marginal Optimal Transport problem with the given cost
    matrix of shape (m_1, ..., m_K) and measure weights (b_1, ..., b_K) with b_k
    in the m_k simplex.

    Returns an optimal coupling coupling pi of shape (m_1, ..., m_K)
    Adapted from https://github.com/judelo/gmmot/blob/master/python/gmmot.py#L210
    """

    K = len(b_list)
    m_list = [len(b) for b in b_list]

    pi_flat = cvx.Variable(np.prod(m_list))
    constraints = [pi_flat >= 0]
    index = 0
    A = np.zeros((np.sum(m_list), np.prod(m_list)))
    b = np.zeros(np.sum(m_list))
    for i in range(K):
        m = m_list[i]
        b[index:index + m] = b_list[i]
        for k in range(m):
            Ap = np.zeros(m_list)
            tup = ()
            for j in range(K):
                if j == i:
                    tup += (k,)
                else:
                    tup += (slice(0, m_list[j]),)
            Ap[tup] = 1
            A[index + k, :] = Ap.flatten()
        index += m
    constraints += [A @ pi_flat == b]

    objective = cvx.sum(cvx.multiply(M.flatten(), pi_flat))

    prob = cvx.Problem(cvx.Minimize(objective), constraints)
    prob.solve()
    return (pi_flat.value).reshape(*m_list)


def solve_barycentre_multi_marginal(Y_list, b_list, weights, eps=1e-5):
    """"
    Computes the W2 barycentre of the given measures y (m_k, d) with weights w
    using the multi-marginal solver. The output will consider that there is mass
    on a point if the mass is greater than eps / (m_1 * ... * m_K).
    Expects numpy arrays.
    """
    M = multi_marginal_cost_tensor(Y_list, weights)
    pi = solve_MMOT(b_list, M)

    # indices with mass
    m_list = [len(b) for b in b_list]
    indices = np.where(pi > eps / np.prod(m_list))
    n = len(indices[0])  # size of the support of the solution
    a = pi[indices]  # barycentre weights
    # barycentre support
    X = np.zeros((n, d))
    for i, idx_tuple in enumerate(zip(*indices)):
        # y_slice is (Y1[j1], ..., YK[jK]) stacked into shape (K, d)
        y_slice = np.stack([Y_list[k][idx_tuple[k]] for k in range(K)], axis=0)
        X[i] = weights @ y_slice
    a = a / np.sum(a)
    return X, a


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

# %%
import numpy as np
import torch
from ot_bar import TT, TN, solve_NLGWB_GD
from time import time
import matplotlib.pyplot as plt
import ot  # type: ignore


device = 'cuda' if torch.cuda.is_available() else 'cpu'
np.random.seed(42)
torch.manual_seed(42)
# number of points of the original measure
K = 100

# dimensions of the original measure
d = 2
L = K  # number of points of the optimised barycentre

# weights of the two measures
a_list = TT([ot.unif(K)] * 4)

# weights for the barycentre
weights = TT(ot.unif(K))


# map 1: R^2 -> R^2 projection onto circle
def proj_circle(X: torch.tensor, origin: torch.tensor, radius: float):
    diffs = X - origin[None, :]
    norms = torch.norm(diffs, dim=1)
    return origin[None, :] + radius * diffs / norms[:, None]


# X_square = TT(torch.rand(K, d))
t = np.linspace(0, 2 * np.pi, K, endpoint=False)
X_square = .5 * TT(torch.tensor([np.cos(t), np.sin(t)]).T)
X_square = X_square + TT(torch.tensor([.5, .5]))[None, :]
origin1 = TT(torch.tensor([-1, -1]))
origin2 = TT(torch.tensor([-1, 2]))
origin3 = TT(torch.tensor([2, 2]))
origin4 = TT(torch.tensor([2, -1]))
r = np.sqrt(2)
P_list = [lambda X: proj_circle(X, origin1, r),
          lambda X: proj_circle(X, origin2, r),
          lambda X: proj_circle(X, origin3, r),
          lambda X: proj_circle(X, origin4, r)]

# positions of the two measures: for a simple optimisation problem, we take
# X1 as the image of a [0, 1]^2 uniform measure by the map P1,
# and X2 as the image of a [0, 1]^2 uniform measure by the map P2,
# the target Y should look uniform [0, 1]^2.
X_list = [P(X_square) for P in P_list]

# %%
# optimiser parameters
learning_rate = .1  # initial learning rate
its = 500  # Gradient Descent iterations
stop_threshold = 1e-20  # stops if |loss_{t+1} - loss_{t}| < this
gamma = 1  # learning rate at step t is initial learning rate * gamma^t

t0 = time()
Y, b, loss_list, exit_status = solve_NLGWB_GD(X_list, a_list, weights,
                                              P_list, L, d, return_exit_status=True, eta_init=learning_rate,
                                              its=its, stop_threshold=stop_threshold,
                                              gamma=gamma)
dt = time() - t0
print(f"Finished in {dt:.2f}s, exit status: {exit_status}, final loss: {loss_list[-1]:.2f}")

# %%
alpha = .5
labels = ['circle 1', 'circle 2', 'circle 3', 'circle 4']
for X, label in zip(X_list, labels):
    plt.scatter(*TN(X).T, alpha=alpha, label=label)
plt.scatter(*TN(X_square).T, label='original', c='gray', alpha=alpha)
plt.scatter(*TN(Y).T, label='GWB', c='black', alpha=alpha)

# %%
plt.plot(loss_list)
plt.yscale('log')
# %%

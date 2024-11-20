# %%
import numpy as np
import torch
from ot_bar.solvers import solve_NLGWB_GD, solve_OT_barycenter_fixed_point
from ot_bar.utils import TT, TN
from time import time
import matplotlib.pyplot as plt
import ot  # type: ignore
from torch.optim import SGD
import matplotlib.animation as animation


device = 'cuda' if torch.cuda.is_available() else 'cpu'

n = 100  # number of points of the original measure and of the barycentre
d = 2  # dimensions of the original measure
K = 4  # number of measures to barycentre
a_list = TT([ot.unif(n)] * K)  # weights of the 4 measures
weights = TT(ot.unif(n))  # weights for the barycentre


# map 1: R^2 -> R^2 projection onto circle
def proj_circle(X: torch.tensor, origin: torch.tensor, radius: float):
    diffs = X - origin[None, :]
    norms = torch.norm(diffs, dim=1)
    return origin[None, :] + radius * diffs / norms[:, None]


# build a measure as a 2D circle
t = np.linspace(0, 2 * np.pi, n, endpoint=False)
X = .5 * TT(torch.tensor([np.cos(t), np.sin(t)]).T)
X = X + TT(torch.tensor([.5, .5]))[None, :]
origin1 = TT(torch.tensor([-1, -1]))
origin2 = TT(torch.tensor([-1, 2]))
origin3 = TT(torch.tensor([2, 2]))
origin4 = TT(torch.tensor([2, -1]))
r = np.sqrt(2)
P_list = [lambda X: proj_circle(X, origin1, r),
          lambda X: proj_circle(X, origin2, r),
          lambda X: proj_circle(X, origin3, r),
          lambda X: proj_circle(X, origin4, r)]

# measures to barycentre are projections of the circle onto other circles
Y_list = [P(X) for P in P_list]

# %% Find generalised barycenter using gradient descent
# optimiser parameters
learning_rate = 30  # initial learning rate
its = 2000  # Gradient Descent iterations
stop_threshold = 1e-20  # stops if |loss_{t+1} - loss_{t}| < this
gamma = 1  # learning rate at step t is initial learning rate * gamma^t
np.random.seed(42)
torch.manual_seed(42)
t0 = time()
X_bar, b, loss_list, exit_status = solve_NLGWB_GD(Y_list, a_list, weights,
                                                  P_list, n, d, return_exit_status=True, eta_init=learning_rate,
                                                  its=its, stop_threshold=stop_threshold,
                                                  gamma=gamma)
dt = time() - t0
print(f"Finished in {dt:.2f}s, exit status: {exit_status}, final loss: {loss_list[-1]:.2f}")

# %% Plot GD barycentre
alpha = .5
labels = ['circle 1', 'circle 2', 'circle 3', 'circle 4']
for Y, label in zip(Y_list, labels):
    plt.scatter(*TN(Y).T, alpha=alpha, label=label)
plt.scatter(*TN(X).T, label='original', c='gray', alpha=alpha)
plt.scatter(*TN(X_bar).T, label='GWB', c='black', alpha=alpha)
plt.axis('equal')
plt.axis('off')
plt.xlim(-.3, 1.3)
plt.ylim(-.3, 1.3)
plt.legend(loc='upper right')
plt.savefig('gwb_circles_gd.pdf')

# %% Plot GD barycentre loss
plt.plot(loss_list)
plt.yscale('log')
plt.savefig('gwb_circles_gd_loss.pdf')

# %% Solve with fixed-point iterations: studying the energy for the function B

# cost_list[k] is a function taking x (n, d) and y (n_k, d_k) and returning a
# (n, n_k) array of costs
cost_list = [lambda x, y: ot.dist(P(x) - y) for P in P_list]


def C(x, y):
    """
    Computes the barycenter cost for candidate points x (n, d) and
    measure supports y: List(n, d_k).
    """
    n = x.shape[0]
    K = len(y)
    out = torch.zeros(n, device=device)
    for k in range(K):
        out += (1 / K) * torch.sum((P_list[k](x) - y[k])**2, axis=1)
    return out


n_vis = 100
u = torch.linspace(-.3, 1.3, n_vis, device=device)
uu, vv = torch.meshgrid(u, u)
x = torch.stack([uu.flatten(), vv.flatten()], dim=1)
y_idx = [n // 5, 2 * n // 5, 3 * n // 5, 4 * n // 5]
y = [(Y_list[k][y_idx[k]])[None, :] * torch.ones_like(x, device=device) for k in range(4)]
M = C(x, y)  # shape (n_vis**2)
M = TN(M.reshape(n_vis, n_vis))
plt.imshow(M.T, interpolation="nearest", origin="lower", cmap='gray')
plt.savefig('B_energy_map.pdf')

# %% define B using GD on its energy
np.random.seed(42)
torch.manual_seed(42)


def B(y, its=100, lr=1, log=False):
    """
    Computes the barycenter images for candidate points x (n, d) and
    measure supports y: List(n, d_k).
    Output: (n, d) array
    """
    x = torch.randn(n, d, device=device, dtype=torch.double)
    x.requires_grad_(True)
    loss_list = []
    opt = SGD([x], lr=lr)
    for _ in range(its):
        opt.zero_grad()
        loss = torch.sum(C(x, y))
        loss.backward()
        opt.step()
        loss_list.append(loss.item())
    if log:
        return x, loss_list
    return x


Bx, loss_list = B(Y_list, its=250, lr=1, log=True)
plt.plot(loss_list)
plt.yscale('log')
plt.savefig('gwb_circles_B_loss.pdf')


# %% Use the fix-point algorithm
np.random.seed(0)
torch.manual_seed(0)

fixed_point_its = 15
X_init = torch.rand(n, d, device=device, dtype=torch.double)
X_bar, X_bar_list = solve_OT_barycenter_fixed_point(X_init, Y_list, cost_list,
                                                    B, max_its=fixed_point_its, pbar=True, log=True)

# %% plot fixed-point barycentre final step
for Y, label in zip(Y_list, labels):
    plt.scatter(*TN(Y).T, alpha=alpha, label=label)
plt.scatter(*TN(X).T, label='original', c='gray', alpha=alpha)
plt.scatter(*TN(X_bar_list[-1]).T, label='GWB', c='black', alpha=alpha)
plt.axis('equal')
plt.xlim(-.3, 1.3)
plt.ylim(-.3, 1.3)
plt.axis('off')
plt.legend()
plt.savefig('gwb_circles_fixed_point.pdf')

# %% animate fixed-point barycentre steps
num_frames = fixed_point_its + 1  # +1 for initialisation
fig, ax = plt.subplots()
ax.set_xlim(-.3, 1.3)
ax.set_ylim(-.3, 1.3)
ax.set_title("Fixed-Point Barycenter Iterations Animation")
ax.axis('equal')
ax.axis('off')

for Y, label in zip(Y_list, labels):
    ax.scatter(*TN(Y).T, alpha=alpha, label=label)
ax.scatter(*TN(X).T, label='original', c='gray', alpha=alpha)

# Plot moving scatter points (initialized empty)
moving_scatter = ax.scatter([], [], color='black', label="GWB", alpha=alpha)

ax.legend(loc="upper right")


def update(frame):  # Update function for animation
    # Update moving scatterplot data
    moving_scatter.set_offsets(TN(X_bar_list[frame]))
    return moving_scatter,


ani = animation.FuncAnimation(fig, update, frames=num_frames, blit=True)
ani.save("fixed_point_barycentre_animation.gif", writer="pillow", fps=2)

# %% First 5 steps on a subplot
fig, axes = plt.subplots(1, 5, figsize=(15, 3))  # 1 row, 5 columns
fig.suptitle("First 5 Steps Fixed-point GWB solver", fontsize=16)

for i, ax in enumerate(axes):
    for Y, label in zip(Y_list, labels):
        ax.scatter(*TN(Y).T, alpha=alpha, label=label)
    ax.scatter(*TN(X).T, label='original', c='gray', alpha=alpha)
    ax.scatter(*TN(X_bar_list[i]).T, label='GWB', c='black', alpha=alpha)
    ax.axis('equal')
    ax.axis('off')
    ax.set_xlim(-.3, 1.3)
    ax.set_ylim(-.3, 1.3)
    ax.set_title(f"Step {i+1}", y=-0.2)
plt.savefig('gwb_circles_fixed_point_5_steps.pdf')

# %%

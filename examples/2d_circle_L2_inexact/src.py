# %%
import numpy as np
import torch
from ot_bar.solvers import solve_OT_barycenter_fixed_point, solve_OT_barycenter_GD, StoppingCriterionReached
from ot_bar.utils import TT, TN
import matplotlib.pyplot as plt
from time import time
import ot  # type: ignore
from torch.optim import Adam
import matplotlib.animation as animation
from matplotlib import cm


np.random.seed(42)
torch.manual_seed(42)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

n = 100  # number of points of the of the barycentre
d = 2  # dimensions of the original measure
K = 4  # number of measures to barycentre
m = 50  # number of points of the measures
b_list = TT([ot.unif(m)] * K)  # weights of the 4 measures
weights = TT(ot.unif(K))  # weights for the barycentre


# map R^2 -> R^2 projection onto circle
def proj_circle(X: torch.tensor, origin: torch.tensor, radius: float):
    diffs = X - origin[None, :]
    norms = torch.norm(diffs, dim=1)
    return origin[None, :] + radius * diffs / norms[:, None]


# circles on which to project
origin1 = TT(torch.tensor([-1, -1]))
origin2 = TT(torch.tensor([-1, 2]))
origin3 = TT(torch.tensor([2, 2]))
origin4 = TT(torch.tensor([2, -1]))
r = np.sqrt(2)
P_list = [lambda X: proj_circle(X, origin1, r),
          lambda X: proj_circle(X, origin2, r),
          lambda X: proj_circle(X, origin3, r),
          lambda X: proj_circle(X, origin4, r)]

# measures to barycentre are projections of different random circles
# onto other circles
Y_list = []
for k in range(K):
    t = np.random.rand(m) * 2 * np.pi
    X_temp = .5 * TT(torch.tensor([np.cos(t), np.sin(t)]).T)
    X_temp = X_temp + TT(torch.tensor([.5, .5]))[None, :]
    Y_list.append(P_list[k](X_temp))


# cost_list[k] is a function taking x (n, d) and y (n_k, d_k) and returning a
# (n, n_k) matrix of costs
def c1(x, y):
    return ot.dist(P_list[0](x), y)


def c2(x, y):
    return ot.dist(P_list[1](x), y)


def c3(x, y):
    return ot.dist(P_list[2](x), y)


def c4(x, y):
    return ot.dist(P_list[3](x), y)


cost_list = [c1, c2, c3, c4]
# %% Find generalised barycenter using gradient descent
# optimiser parameters
learning_rate = 30  # initial learning rate
its = 2000  # Gradient Descent iterations
stop_threshold = 1e-10  # stops if W2^2(X_{t+1}, X_{t}) / |X_t| < this
gamma = 1  # learning rate at step t is initial learning rate * gamma^t
np.random.seed(42)
torch.manual_seed(42)
t0 = time()
X_bar, a, log_dict = solve_OT_barycenter_GD(
    Y_list, b_list, weights, cost_list, n, d, eta_init=learning_rate, its=its, stop_threshold=stop_threshold, gamma=gamma, log=True)
dt = time() - t0
print(f"Finished in {dt:.2f}s, exit status: {log_dict['exit_status']}, final loss: {log_dict['loss_list'][-1]:.10f}")

# %% Plot GD barycentre
alpha = .5
labels = ['circle 1', 'circle 2', 'circle 3', 'circle 4']
for Y, label in zip(Y_list, labels):
    plt.scatter(*TN(Y).T, alpha=alpha, label=label)
plt.scatter(*TN(X_bar).T, label='GWB', c='black', alpha=alpha)
plt.axis('equal')
plt.axis('off')
plt.xlim(-.3, 1.3)
plt.ylim(-.3, 1.3)
plt.legend(loc='upper right')
plt.savefig('gwb_circles_gd.pdf')

# %% Plot GD barycentre loss
plt.plot(log_dict['loss_list'])
plt.yscale('log')
plt.savefig('gwb_circles_gd_loss.pdf')


# %% Solve with fixed-point iterations: studying the energy for the function B
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
y_idx = [m // 5, 2 * m // 5, 3 * m // 5, 4 * m // 5]
y = [(Y_list[k][y_idx[k]])[None, :] * torch.ones_like(x, device=device) for k in range(4)]
M = C(x, y)  # shape (n_vis**2)
M = TN(M.reshape(n_vis, n_vis))
uu, vv = TN(uu), TN(vv)

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
surf = ax.plot_surface(uu, vv, M, cmap=cm.CMRmap, linewidth=0,
                       antialiased=False)
for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
    axis.set_ticklabels([])
plt.savefig("B_energy_map.pdf", format="pdf", bbox_inches="tight")
plt.show(block=True)

# %% define B using GD on its energy
np.random.seed(42)
torch.manual_seed(42)


def B(y, its=150, lr=1, log=False, stop_threshold=stop_threshold):
    """
    Computes the barycenter images for measure supports y: List(n, d_k).
    Output: (n, d) array
    """
    x = torch.randn(n, d, device=device, dtype=torch.double)
    x.requires_grad_(True)
    loss_list = [1e10]
    opt = Adam([x], lr=lr)
    exit_status = 'unknown'
    try:
        for _ in range(its):
            x_prev = x.data.clone()
            opt.zero_grad()
            loss = torch.sum(C(x, y))
            loss.backward()
            opt.step()
            loss_list.append(loss.item())
            diff = torch.sum((x.data - x_prev)**2)
            current = torch.sum((x_prev)**2)
            if diff / current < stop_threshold:
                exit_status = 'Local optimum'
                raise StoppingCriterionReached
        exit_status = 'Max iterations reached'
        raise StoppingCriterionReached
    except StoppingCriterionReached:
        if log:
            return x, {'loss_list': loss_list[1:], 'exit_status': exit_status}
        return x


X_init = torch.rand(n, d, device=device, dtype=torch.double)
a = TT(ot.unif(n))
b = TT(ot.unif(m))
pi_list = [ot.emd(a, b, cost_list[k](X_init, Y_list[k])) for k in range(K)]
Y_perm = []
for k in range(K):
    Y_perm.append(n * pi_list[k] @ Y_list[k])
Bx, log = B(Y_perm, its=150, lr=1, log=True)
plt.plot(log['loss_list'])
plt.yscale('log')
plt.savefig('gwb_circles_B_loss.pdf')


# %% Use the fixed-point algorithm
np.random.seed(0)
torch.manual_seed(0)

t0 = time()
fixed_point_its = 15
X_init = torch.rand(n, d, device=device, dtype=torch.double)
b_list = [TT(ot.unif(m))] * K
X_bar, log_dict = solve_OT_barycenter_fixed_point(
    X_init, Y_list, b_list, cost_list, B, max_its=fixed_point_its, pbar=True, log=True, stop_threshold=stop_threshold)
dt = time() - t0
print(f"Finished in {dt:.2f}s, exit status: {log_dict['exit_status']}")

# %% plot fixed-point barycentre final step
alpha = .5
labels = ['circle 1', 'circle 2', 'circle 3', 'circle 4']
for Y, label in zip(Y_list, labels):
    plt.scatter(*TN(Y).T, alpha=alpha, label=label)
plt.scatter(*TN(log_dict['X_list'][-1]).T, label='GWB', c='black', alpha=alpha)
plt.axis('equal')
plt.xlim(-.3, 1.3)
plt.ylim(-.3, 1.3)
plt.axis('off')
plt.legend()
plt.tight_layout()
plt.savefig('gwb_circles_fixed_point.pdf', format='pdf', bbox_inches='tight')

# %% animate fixed-point barycentre steps
num_frames = len(log_dict['X_list'])
fig, ax = plt.subplots()
ax.set_xlim(-.3, 1.3)
ax.set_ylim(-.3, 1.3)
ax.set_title("Fixed-Point Barycenter Iterations Animation")
ax.axis('equal')
ax.axis('off')

for Y, label in zip(Y_list, labels):
    ax.scatter(*TN(Y).T, alpha=alpha, label=label)

# Plot moving scatter points (initialized empty)
moving_scatter = ax.scatter([], [], color='black', label="GWB", alpha=alpha)

ax.legend(loc="upper right")


def update(frame):  # Update function for animation
    # Update moving scatterplot data
    moving_scatter.set_offsets(TN(log_dict['X_list'][frame]))
    return moving_scatter,


ani = animation.FuncAnimation(fig, update, frames=num_frames, blit=True)
ani.save("fixed_point_barycentre_animation.gif", writer="pillow", fps=2)

# %% First 5 steps on a subplot
n_plots = 5
fig, axes = plt.subplots(1, n_plots, figsize=(3 * n_plots, 3))
fig.suptitle(f"First {n_plots} Steps Fixed-point GWB solver", fontsize=16)

for i, ax in enumerate(axes):
    for Y, label in zip(Y_list, labels):
        ax.scatter(*TN(Y).T, alpha=alpha, label=label)
    ax.scatter(*TN(log_dict['X_list'][i]).T, label='GWB', c='black', alpha=alpha)
    ax.axis('equal')
    ax.axis('off')
    ax.set_xlim(-.3, 1.3)
    ax.set_ylim(-.3, 1.3)
    ax.set_title(f"Step {i+1}", y=-0.2)

plt.savefig(f'gwb_circles_fixed_point_{n_plots}_steps.pdf',
            format='pdf', bbox_inches='tight')

# %% Barycentre energy for fixed-point iterations
V_list = []
a = TT(ot.unif(n))
b = TT(ot.unif(m))
for X in log_dict['X_list']:
    V = 0
    for k in range(K):
        V += (1 / K) * ot.emd2(a, b, ot.dist(P_list[k](X), Y_list[k]))
    V_list.append(V.item())

plt.plot(V_list, linewidth=5, alpha=.8, color='#4A90E2')
plt.title('V evolution by iteration')
plt.xlabel('Iteration')
plt.ylabel('V')
plt.yscale('log')
plt.tight_layout()
plt.savefig('gwb_circles_fixed_point_V.pdf')

# %%

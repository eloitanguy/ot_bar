# %%
import numpy as np
from ot_bar.solvers import solve_OT_barycenter_fixed_point
from ot_bar.utils import imageToGrid
import matplotlib.pyplot as plt
import ot  # type: ignore
import matplotlib.animation as animation


K = 3
d = 3
P1 = np.array([[1, 0, 0], [0, 1, 0]])
P2 = np.array([[0, 1, 0], [0, 0, 1]])
P3 = np.array([[1, 0, 0], [0, 0, 1]])
P_list = [P1, P2, P3]

c_list = ['#7ED321', '#4A90E2', '#9013FE']
c_bar = '#D0021B'

# first dataset should be the one with the smallest nb of points
spider = plt.imread('spider.png')
spider = 1 - spider[::4, ::4, 0]
Y1 = imageToGrid(spider, constant_threshold=False)
Y1 = np.concatenate((Y1, Y1, Y1), axis=0)  # increase nb of points in Y1
n = Y1.shape[0]  # number of points in all datasets
Y1 = Y1 - np.array([.5, .5])[None, :]  # centre

am = plt.imread('america.png')
am = 1 - am[::4, ::4, 0]
Y2 = imageToGrid(am, constant_threshold=False)
Y2 = np.concatenate((Y2, Y2), axis=0)  # increase nb of points in Y1
# subsampling of the data to match number of points in Y1
indices = np.random.choice(Y2.shape[0], n, replace=False)
Y2 = Y2[indices, :]
Y2 = Y2 - np.array([.5, .5])[None, :]  # centre

h = plt.imread('batman.png')
h = 1 - h[::4, ::4, 0]
Y3 = imageToGrid(h, constant_threshold=False)
# subsampling of the data to match number of points in Y1
indices = np.random.choice(Y3.shape[0], n, replace=False)
Y3 = Y3[indices, :]
Y3 = Y3 - np.array([.5, .5])[None, :]  # centre

Y_list = [Y1, Y2, Y3]
weights = ot.unif(K)
b_list = [ot.unif(n) for _ in range(K)]


def cost1(X, Y):
    X_proj = X @ P1.T
    return np.linalg.norm(X_proj[:, None, :] - Y[None, :, :], axis=-1)


def cost2(X, Y):
    X_proj = X @ P2.T
    return np.linalg.norm(X_proj[:, None, :] - Y[None, :, :], axis=-1)


def cost3(X, Y):
    X_proj = X @ P3.T
    return np.linalg.norm(X_proj[:, None, :] - Y[None, :, :], axis=-1)


cost_list = [cost1, cost2, cost3]
T = 1


def B(Y_list):
    for _ in range(T):
        A = np.zeros((d, d))
        Y_embeddings = np.zeros((n, d))
        Y_embeddings_copy = np.copy(Y_embeddings)
        for k in range(len(Y_list)):
            dk = np.linalg.norm(Y_list[k] - Y_embeddings_copy @ P_list[k].T)
            Y_embeddings += weights[k] * Y_list[k] @ P_list[k] / dk
            A += weights[k] * P_list[k].T @ P_list[k] / dk
        Y_embeddings = Y_embeddings @ np.linalg.inv(A)
    return Y_embeddings


# %% compute barycenter using the fixed point algorithm
np.random.seed(0)
X0 = np.random.rand(n, d)
X, log_dict = solve_OT_barycenter_fixed_point(
    X0, Y_list, b_list, cost_list, B, max_its=30, pbar=True, log=True)

# %%
Y1_3D = Y1 @ P1
Y2_3D = Y2 @ P2
Y3_3D = Y3 @ P3
Y_3D_list = [Y1_3D, Y2_3D, Y3_3D]
offsets = [
    1 * np.array([0, 0, 1]),
    1 * np.array([1, 0, 0]),
    1 * np.array([0, 1, 0])
]
Y_3D_list = [Y - offset for Y, offset in zip(Y_3D_list, offsets)]


fig = plt.figure(figsize=(7, 7))
axis = fig.add_subplot(1, 1, 1, projection="3d")

axis.scatter(X[:, 0], X[:, 1], X[:, 2], alpha=.3,
             c=c_bar, marker='o')

for k in range(K):
    axis.scatter(Y_3D_list[k][:, 0], Y_3D_list[k][:, 1], Y_3D_list[k][:, 2],
                 alpha=.3, c=c_list[k], marker='o')

axis.view_init(elev=30, azim=60)
axis.set_xticks([])
axis.set_yticks([])
axis.set_zticks([])
plt.tight_layout()
plt.savefig("bar_3D.pdf", format="pdf", bbox_inches="tight")
plt.show()

# %%
fig, axs = plt.subplots(2, 3, figsize=(15, 10))

for k in range(K):
    X_proj = X @ P_list[k].T
    axs[0, k].scatter(Y_list[k][:, 0], Y_list[k][:, 1], alpha=.5,
                      c=c_list[k], marker='o')
    axs[0, k].axis('equal')
    axs[0, k].axis('off')
    axs[0, k].set_title(f"Measure {k + 1}")
    axs[1, k].scatter(X_proj[:, 0], X_proj[:, 1], alpha=.5,
                      c=c_bar, marker='o')
    axs[1, k].axis('equal')
    axs[1, k].axis('off')
    axs[1, k].set_title("Projected Barycentre", y=-.1)
plt.savefig("bar_proj.pdf", format="pdf", bbox_inches="tight")
plt.show()

# %% animation
# Send the input measures into 3D space for visualization
Y_visu = Y_3D_list

fig = plt.figure(figsize=(7, 7))
ax = fig.add_subplot(1, 1, 1, projection="3d")


def _init():
    for Yi, c in zip(Y_visu, c_list):
        ax.scatter(Yi[:, 0], Yi[:, 1], Yi[:, 2], marker='o', alpha=.3, c=c)
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], marker='o', alpha=.3, c=c_bar)
    ax.view_init(elev=0, azim=0)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    return fig,


def _update_plot(i):
    if i < 45:
        ax.view_init(elev=0, azim=4 * i)
    else:
        ax.view_init(elev=i - 45, azim=4 * i)
    return fig,


ani = animation.FuncAnimation(fig, _update_plot, init_func=_init, frames=136, interval=100, blit=True, repeat_delay=2000)
ani.save("barycenter_rotation.gif", writer="pillow", fps=10)

# %%  Point-cloud distances between steps
plt.clf()
one_step_diffs = []
a = ot.unif(n)
for i in range(1, len(log_dict["X_list"])):
    one_step_diffs.append(
        ot.emd2(a, a, ot.dist(
            (log_dict["X_list"])[i - 1], (log_dict["X_list"])[i])))

plt.plot(one_step_diffs)
plt.xlabel("Iteration")
plt.ylabel("$W_2^2$")
plt.title("W2 distance between consecutive steps")
plt.savefig("W2_consecutive_steps.pdf", format="pdf", bbox_inches="tight")

# %%

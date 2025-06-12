# %%
import numpy as np
import matplotlib.pyplot as plt
import ot  # type: ignore
import cv2
import torch
from torch.optim import Adam
from ot_bar.utils import TT, TN
from ot_bar.solvers import solve_OT_barycenter_fixed_point, StoppingCriterionReached


IA1 = plt.imread('A1.jpg').astype(np.float64) / 255
IB1o = plt.imread('B1o.jpg').astype(np.float64) / 255
IB2 = plt.imread('B2.jpg').astype(np.float64) / 255
IC1 = plt.imread('C1.jpg').astype(np.float64) / 255
n = IA1.shape[0] * IA1.shape[1]
plt.imshow(IA1)
plt.axis('off')

# %% downsize images + define costs and B
w_downscale = 50
IA1d = cv2.resize(IA1, (w_downscale, w_downscale))
plt.imsave('A1d.jpg', IA1d)
IB1od = cv2.resize(IB1o, (w_downscale, w_downscale))
plt.imsave('B1od.jpg', IB1od)
IB2d = cv2.resize(IB2, (w_downscale, w_downscale))
plt.imsave('B2d.jpg', IB2d)
IC1d = cv2.resize(IC1, (w_downscale, w_downscale))
plt.imsave('C1d.jpg', IC1d)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
nd = w_downscale ** 2
K = 3
d = 3
Y_list = [TT(IA1d.reshape(nd, 3)),
          TT(IB1od.reshape(nd, 3)),
          TT(IB2d.reshape(nd, 3))]
b_list = TT([ot.unif(nd) for _ in range(3)])
X_init = Y_list[-1].clone()


def p_norm_q_cost_matrix(u, v, p, q):
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


def B(y, p, q, its=200, lr=1e-2, log=False, stop_threshold=1e-20):
    """
    Computes the barycenter images for candidate points x (n, d) and
    measure supports y: List(n, d_k).
    Output: (n, d) array
    """
    x = torch.rand(nd, d, device=device, dtype=torch.double)
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


# %% compute 1-norm barycenter
torch.manual_seed(42)
p, q = 1., 1.
cost_list = [lambda x, y: p_norm_q_cost_matrix(x, y, p, q)] * K
its = 3
X_bar, log_dict = solve_OT_barycenter_fixed_point(
    X_init, Y_list, b_list, cost_list, lambda y: B(y, p, q),
    max_its=its, log=True, stop_threshold=0.)

# %% apply barycenter colour distribution to C1d
X_bar_np = TN(X_bar)
X_bar_np = np.clip(X_bar_np, 0., 1.)
pi = ot.emd(ot.unif(nd), ot.unif(nd),
            ot.dist(IC1d.reshape(nd, 3), X_bar_np))
IC1d_matched_bar = np.reshape(
    nd * pi @ X_bar_np,
    IA1d.shape)
plt.imsave('C1d_matched_bar.jpg', IC1d_matched_bar)
plt.imshow(IC1d_matched_bar)

# %% apply the matching to the true-scale image via interpolation
# apply the linear transformation of the closest RBG vector in the downscaled
# version
IC1d_flat = IC1d.reshape(nd, 3)
IC1dm_flat = IC1d_matched_bar.reshape(nd, 3)
IC1_flat = IC1.reshape(n, 3)
# closest pixel in downscaled image
pixel_idx = np.argmin(ot.dist(IC1_flat, IC1d_flat), axis=1)
translations = IC1dm_flat - IC1d_flat
IC1m_flat = IC1_flat + translations[pixel_idx]
IC1m_flat = np.clip(IC1m_flat, 0., 1.)
IC1m = np.reshape(IC1m_flat, IC1.shape)
plt.imsave('C1_matched_bar.jpg', IC1m)
plt.imshow(IC1m)

# %% compute W2 barycenter
p, q = 2, 2
cost_list = [lambda x, y: p_norm_q_cost_matrix(x, y, p, q)] * K
its = 3
X_bar2, log_dict2 = solve_OT_barycenter_fixed_point(
    X_init, Y_list, b_list, cost_list, lambda y: B(y, p, q),
    max_its=its, log=True, stop_threshold=0.)

# %% apply W2 barycenter colour distribution to C1d
X_bar2_np = TN(X_bar2)
X_bar2_np = np.clip(X_bar2_np, 0., 1.)
pi = ot.emd(ot.unif(nd), ot.unif(nd),
            ot.dist(IC1d.reshape(nd, 3), X_bar2_np))
IC1d_matched_bar2 = np.reshape(
    nd * pi @ X_bar2_np,
    IA1d.shape)
plt.imsave('C1d_matched_bar2.jpg', IC1d_matched_bar2)
plt.imshow(IC1d_matched_bar2)

# %% apply the matching to the true-scale image via interpolation
# apply the linear transformation of the closest RBG vector in the downscaled
# version
IC1d_flat = IC1d.reshape(nd, 3)
IC1dm2_flat = IC1d_matched_bar2.reshape(nd, 3)
IC1_flat = IC1.reshape(n, 3)
# closest pixel in downscaled image
pixel_idx = np.argmin(ot.dist(IC1_flat, IC1d_flat), axis=1)
translations = IC1dm2_flat - IC1d_flat
IC1m2_flat = IC1_flat + translations[pixel_idx]
IC1m2_flat = np.clip(IC1m2_flat, 0., 1.)
IC1m2 = np.reshape(IC1m2_flat, IC1.shape)
plt.imsave('C1_matched_bar2.jpg', IC1m2)
plt.imshow(IC1m2)

# %% show images
n_im = 6
n_rows = 2
size = 3
fig, axes = plt.subplots(n_rows, n_im // n_rows,
                         figsize=((n_im // n_rows) * size, size * n_rows))

IA1 = plt.imread('A1.jpg').astype(np.float64) / 255
axes[0, 0].imshow(IA1)
axes[0, 0].axis('off')
axes[0, 0].set_title('Source 1')

IB1o = plt.imread('B1o.jpg').astype(np.float64) / 255
axes[0, 1].imshow(IB1o)
axes[0, 1].axis('off')
axes[0, 1].set_title('Source 2 (outlier)')

IB2 = plt.imread('B2.jpg').astype(np.float64) / 255
axes[0, 2].imshow(IB2)
axes[0, 2].axis('off')
axes[0, 2].set_title('Source 3')

IC1 = plt.imread('C1.jpg').astype(np.float64) / 255
axes[1, 0].imshow(IC1)
axes[1, 0].axis('off')
axes[1, 0].set_title('Input')

IC1m = plt.imread('C1_matched_bar.jpg').astype(np.float64) / 255
axes[1, 1].imshow(IC1m)
axes[1, 1].axis('off')
axes[1, 1].set_title('Input matched to W1 bar')

IC1m2 = plt.imread('C1_matched_bar2.jpg').astype(np.float64) / 255
axes[1, 2].imshow(IC1m2)
axes[1, 2].axis('off')
axes[1, 2].set_title('Input matched to W2 bar')
plt.tight_layout()
plt.savefig('bar_W1_W2_matching.jpg', format='jpg', dpi=300)
plt.show()


# %% show colours
def scatter_rgb(img_path, ax):
    im = plt.imread(img_path).astype(np.float64) / 255
    pix = im.reshape(-1, 3)
    ax.scatter(pix[:, 0], pix[:, 1], pix[:, 2],
               c=pix, marker="o", alpha=0.3)


size = 3
n_rows = 2
n_cols = 4
fig, axes = plt.subplots(n_rows, n_cols,
                         figsize=(n_cols * size, n_rows * size),
                         subplot_kw={'projection': '3d'})

scatter_rgb('A1d.jpg', axes[0, 0])
axes[0, 0].set_title('Source 1')

scatter_rgb('B1od.jpg', axes[0, 1])
axes[0, 1].set_title('Source 2 (outlier)')

scatter_rgb('B2d.jpg', axes[0, 2])
axes[0, 2].set_title('Source 3')

axes[0, 3].scatter(*X_bar_np.T, c=X_bar_np, marker="o", alpha=0.3)
axes[0, 3].set_title('W1 Barycenter')

axes[1, 0].scatter(*X_bar2_np.T, c=X_bar2_np, marker="o", alpha=0.3)
axes[1, 0].set_title('W2 Barycenter')

scatter_rgb('C1d.jpg', axes[1, 1])
axes[1, 1].set_title('Input')

scatter_rgb('C1d_matched_bar.jpg', axes[1, 2])
axes[1, 2].set_title('Input matched to W1 bar')

scatter_rgb('C1d_matched_bar2.jpg', axes[1, 3])
axes[1, 3].set_title('Input matched to W2 bar')

for ax in axes.flat:
    ax.axis('off')

plt.tight_layout()
plt.savefig('bar_W1_W2_matching_rgb.jpg', format='jpg', dpi=300)
plt.show()

# %%

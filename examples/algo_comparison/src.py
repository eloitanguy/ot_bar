# %%
import numpy as np
import torch
from ot_bar.solvers import solve_w2_barycentre_multi_marginal, solve_OT_barycenter_fixed_point, solve_OT_barycenter_GD
from ot_bar.utils import TT, TN, plot_runs
from time import time
import matplotlib.pyplot as plt
import ot  # type: ignore
from tqdm import tqdm
import json
import os


device = 'cpu'
colours = ['#7ED321', '#4A90E2', '#9013FE']


def V(X, a, Y_list, b_list, weights):
    v = 0
    for k in range(len(Y_list)):
        v += weights[k] * ot.emd2(a, b_list[k], ot.dist(X, Y_list[k]))
    return v


def B(Y, weights):
    """
    Computes the L2 ground barycentre for a list Y of K arrays (n, d) with
    weights w (K,). The output is a (n, d) array.
    """
    X = np.zeros(Y[0].shape)
    for k in range(len(Y)):
        X += weights[k] * Y[k]
    return X


# %% Loops for multiple n and d=10, K=3
seed = 0
np.random.seed(seed)
torch.manual_seed(seed)
d = 10
K = 3
cost_list = [ot.dist] * K
n_list = [10, 30, 50, 70, 100]
fp_its_list = [1, 5, 10, 50]
n_samples = 10
xp_name = f'multiple_n_d{d}_K{K}'
results_file = xp_name + '_results.npy'
params_file = xp_name + '_params.json'

xp_params = {
    'seed': seed,
    'd': d,
    'K': K,
    'n_list': n_list,
    'n_samples': n_samples,
    'stop_threshold': 1e-10,
    'fp_its_list': fp_its_list,
    'gd_its': 1000,
    'gd_eta': 10,
    'gd_gamma': 1,
    'gd_a_unif': True,
    'mm_eps': 1e-5,
    'cost': 'L2'
}


# define the experiments
def run_multiple_n_xp():
    # idx 0 ... len(fp_its_list) - 1 : fp its, idx -1 : mm
    V_results = np.zeros((1 + len(fp_its_list), len(n_list), n_samples))
    dt_results = np.zeros((1 + len(fp_its_list), len(n_list), n_samples))
    iterator = tqdm(n_list)

    for n_idx, n in enumerate(iterator):
        b_list = [ot.unif(n)] * K
        for i in range(n_samples):
            iterator.set_postfix_str(
                f'n={n} [{n_idx + 1}/{len(n_list)}] sample {i + 1}/{n_samples} MM')
            Y_list = []
            for _ in range(K):
                Y_list.append(np.random.randn(n, d))
            X_init = np.random.randn(n, d)
            weights = ot.unif(K)

            # Multi-marginal
            t0 = time()
            X_mm, a_mm = solve_w2_barycentre_multi_marginal(
                Y_list, b_list, weights, eps=xp_params['mm_eps'])
            dt_results[-1, n_idx, i] = time() - t0
            V_results[-1, n_idx, i] = V(X_mm, a_mm, Y_list, b_list, weights)

            # Fixed point
            for fp_idx, fp_its in enumerate(fp_its_list):
                iterator.set_postfix_str(
                    f'n={n} [{n_idx + 1}/{len(n_list)}] sample {i + 1}/{n_samples} FP {fp_its} its [{fp_idx + 1} / {len(fp_its_list)}]')
                t0 = time()
                X_fp = solve_OT_barycenter_fixed_point(
                    X_init, Y_list, b_list, cost_list, lambda y: B(y, weights),
                    max_its=fp_its, stop_threshold=xp_params['stop_threshold'])
                dt_results[fp_idx, n_idx, i] = time() - t0
                a_fp = ot.unif(n)
                V_results[fp_idx, n_idx, i] = V(
                    X_fp, a_fp, Y_list, b_list, weights)

    # write parameters to file
    with open(params_file, 'w') as f:
        json.dump(xp_params, f, indent=4)
    # save results
    np.save(results_file, [V_results, dt_results])


if os.path.exists(results_file) and os.path.exists(params_file):
    # check identical parameters
    with open(params_file, 'r') as f:
        xp_params_loaded = json.load(f)
    if xp_params_loaded != xp_params:
        print(f'Found different parameters, rerunning {xp_name}...')
        run_multiple_n_xp()

if not os.path.exists(results_file) or not os.path.exists(params_file):
    print(f'No results found, running {xp_name}...')
    run_multiple_n_xp()

# load results
V_results, dt_results = np.load(results_file, allow_pickle=True)

# plot results
curve_labels = [f'FP {fp_its} its' for fp_its in fp_its_list[:3]]
V_ratios = V_results[:-1] / V_results[-1][None, :, :]
dt_ratios = dt_results[:-1] / dt_results[-1][None, :, :]
fig, axs = plt.subplots(1, 2, figsize=(6, 3))
plot_runs(V_ratios, x=n_list, ax=axs[0],
          curve_labels=curve_labels, title='V / V MM', x_label='n', x_scale_log=False, y_scale_log=False, legend_loc='lower right',
          curve_colours=colours)
plot_runs(dt_ratios, x=n_list, ax=axs[1],
          curve_labels=curve_labels, title='Time / Time MM', x_label='n', x_scale_log=False, y_scale_log=True, curve_colours=colours)
plt.suptitle('Comparison of FP to MM for different measure sizes n',
             y=1.05, fontsize=14)
plt.subplots_adjust(wspace=0.4) 
plt.savefig(xp_name + '.pdf')
plt.show()

# %% Loops for multiple d and n=30, K=3
seed = 0
np.random.seed(seed)
torch.manual_seed(seed)
n = 30
K = 3
cost_list = [ot.dist] * K
d_list = [10, 30, 50, 70, 100]
fp_its_list = [1, 5, 10, 50]
n_samples = 10
xp_name = f'multiple_d_n{n}_K{K}'
results_file = xp_name + '_results.npy'
params_file = xp_name + '_params.json'

xp_params = {
    'seed': seed,
    'n': n,
    'K': K,
    'd_list': d_list,
    'n_samples': n_samples,
    'stop_threshold': 1e-10,
    'fp_its_list': fp_its_list,
    'gd_its': 1000,
    'gd_eta': 10,
    'gd_gamma': 1,
    'gd_a_unif': True,
    'mm_eps': 1e-5,
    'cost': 'L2'
}


# define the experiments
def run_multiple_d_xp():
    # idx 0 ... len(fp_its_list) - 1 : fp its, idx -1 : mm
    V_results = np.zeros((1 + len(fp_its_list), len(d_list), n_samples))
    dt_results = np.zeros((1 + len(fp_its_list), len(d_list), n_samples))
    iterator = tqdm(d_list)
    b_list = [ot.unif(n)] * K

    for d_idx, d in enumerate(iterator):
        for i in range(n_samples):
            iterator.set_postfix_str(
                f'd={d} [{d_idx + 1}/{len(d_list)}] sample {i + 1}/{n_samples} MM')
            Y_list = []
            for _ in range(K):
                Y_list.append(np.random.randn(n, d))
            X_init = np.random.randn(n, d)
            weights = ot.unif(K)

            # Multi-marginal
            t0 = time()
            X_mm, a_mm = solve_w2_barycentre_multi_marginal(
                Y_list, b_list, weights, eps=xp_params['mm_eps'])
            dt_results[-1, d_idx, i] = time() - t0
            V_results[-1, d_idx, i] = V(X_mm, a_mm, Y_list, b_list, weights)

            # Fixed point
            for fp_idx, fp_its in enumerate(fp_its_list):
                iterator.set_postfix_str(
                    f'd={d} [{d_idx + 1}/{len(d_list)}] sample {i + 1}/{n_samples} FP {fp_its} its [{fp_idx + 1} / {len(fp_its_list)}]')
                t0 = time()
                X_fp = solve_OT_barycenter_fixed_point(
                    X_init, Y_list, b_list, cost_list, lambda y: B(y, weights),
                    max_its=fp_its, stop_threshold=xp_params['stop_threshold'])
                dt_results[fp_idx, d_idx, i] = time() - t0
                a_fp = ot.unif(n)
                V_results[fp_idx, d_idx, i] = V(
                    X_fp, a_fp, Y_list, b_list, weights)

    # write parameters to file
    with open(params_file, 'w') as f:
        json.dump(xp_params, f, indent=4)
    # save results
    np.save(results_file, [V_results, dt_results])


if os.path.exists(results_file) and os.path.exists(params_file):
    # check identical parameters
    with open(params_file, 'r') as f:
        xp_params_loaded = json.load(f)
    if xp_params_loaded != xp_params:
        print(f'Found different parameters, rerunning {xp_name}...')
        run_multiple_d_xp()

if not os.path.exists(results_file) or not os.path.exists(params_file):
    print(f'No results found, running {xp_name}...')
    run_multiple_d_xp()

# load results
V_results, dt_results = np.load(results_file, allow_pickle=True)

# plot results
curve_labels = [f'FP {fp_its} its' for fp_its in fp_its_list[:3]]
V_ratios = V_results[:-1] / V_results[-1][None, :, :]
dt_ratios = dt_results[:-1] / dt_results[-1][None, :, :]
fig, axs = plt.subplots(1, 2, figsize=(6, 3))
plot_runs(V_ratios, x=d_list, ax=axs[0],
          curve_labels=curve_labels, title='V / V MM', x_label='d', x_scale_log=False, y_scale_log=False, curve_colours=colours)
plot_runs(dt_ratios, x=d_list, ax=axs[1],
          curve_labels=curve_labels, title='Time / Time MM', x_label='d', x_scale_log=False, y_scale_log=True, curve_colours=colours)
plt.suptitle('Comparison of FP to MM for different dimensions d',
             y=1.05, fontsize=14)
plt.subplots_adjust(wspace=0.4)
plt.savefig(xp_name + '.pdf')
plt.show()

# %% Loops for multiple K and n=10, d=10
seed = 0
np.random.seed(seed)
torch.manual_seed(seed)
n = 10
d = 10
K_list = [2, 3, 4, 5, 6]
fp_its_list = [1, 5, 10, 50]
n_samples = 10
xp_name = f'multiple_K_n{n}_d{d}'
results_file = xp_name + '_results.npy'
params_file = xp_name + '_params.json'

xp_params = {
    'seed': seed,
    'n': n,
    'd': d,
    'K_list': K_list,
    'n_samples': n_samples,
    'stop_threshold': 1e-10,
    'fp_its_list': fp_its_list,
    'gd_its': 1000,
    'gd_eta': 10,
    'gd_gamma': 1,
    'gd_a_unif': True,
    'mm_eps': 1e-5,
    'cost': 'L2'
}


# define the experiments
def run_multiple_K_xp():
    # idx 0 ... len(fp_its_list) - 1 : fp its, idx -1 : mm
    V_results = np.zeros((1 + len(fp_its_list), len(K_list), n_samples))
    dt_results = np.zeros((1 + len(fp_its_list), len(K_list), n_samples))
    iterator = tqdm(K_list)

    for K_idx, K in enumerate(iterator):
        b_list = [ot.unif(n)] * K
        cost_list = [ot.dist] * K
        for i in range(n_samples):
            iterator.set_postfix_str(
                f'K={K} [{K_idx + 1}/{len(K_list)}] sample {i + 1}/{n_samples} MM')
            Y_list = []
            for _ in range(K):
                Y_list.append(np.random.randn(n, d))
            X_init = np.random.randn(n, d)
            weights = ot.unif(K)

            # Multi-marginal
            t0 = time()
            X_mm, a_mm = solve_w2_barycentre_multi_marginal(
                Y_list, b_list, weights, eps=xp_params['mm_eps'])
            dt_results[-1, K_idx, i] = time() - t0
            V_results[-1, K_idx, i] = V(X_mm, a_mm, Y_list, b_list, weights)

            # Fixed point
            for fp_idx, fp_its in enumerate(fp_its_list):
                iterator.set_postfix_str(
                    f'K={K} [{K_idx + 1}/{len(K_list)}] sample {i + 1}/{n_samples} FP {fp_its} its [{fp_idx + 1} / {len(fp_its_list)}]')
                t0 = time()
                X_fp = solve_OT_barycenter_fixed_point(
                    X_init, Y_list, b_list, cost_list, lambda y: B(y, weights),
                    max_its=fp_its, stop_threshold=xp_params['stop_threshold'])
                dt_results[fp_idx, K_idx, i] = time() - t0
                a_fp = ot.unif(n)
                V_results[fp_idx, K_idx, i] = V(
                    X_fp, a_fp, Y_list, b_list, weights)

    # write parameters to file
    with open(params_file, 'w') as f:
        json.dump(xp_params, f, indent=4)
    # save results
    np.save(results_file, [V_results, dt_results])


if os.path.exists(results_file) and os.path.exists(params_file):
    # check identical parameters
    with open(params_file, 'r') as f:
        xp_params_loaded = json.load(f)
    if xp_params_loaded != xp_params:
        print(f'Found different parameters, rerunning {xp_name}...')
        run_multiple_K_xp()

if not os.path.exists(results_file) or not os.path.exists(params_file):
    print(f'No results found, running {xp_name}...')
    run_multiple_K_xp()

# load results
V_results, dt_results = np.load(results_file, allow_pickle=True)

# plot results
curve_labels = [f'FP {fp_its} its' for fp_its in fp_its_list[:3]]
V_ratios = V_results[:-1] / V_results[-1][None, :, :]
dt_ratios = dt_results[:-1] / dt_results[-1][None, :, :]
fig, axs = plt.subplots(1, 2, figsize=(6, 3))
plot_runs(V_ratios, x=K_list, ax=axs[0],
          curve_labels=curve_labels, title='V / V MM', x_label='K', x_scale_log=False, y_scale_log=False, curve_colours=colours)
plot_runs(dt_ratios, x=K_list, ax=axs[1],
          curve_labels=curve_labels, title='Time / Time MM', x_label='K', x_scale_log=False, y_scale_log=True, curve_colours=colours,
          legend_loc='lower left')
plt.suptitle('Comparison of FP to MM for different components K',
             y=1.05, fontsize=14)
plt.subplots_adjust(wspace=0.4)
plt.savefig(xp_name + '.pdf')
plt.show()

# %% For manual testing
np.random.seed(0)
torch.manual_seed(0)
K = 3
d = 10
n = 20
fp_its = 3
gd_its = 1000
gd_eta = 10
gd_gamma = 1
gd_a_unif = True
stop_threshold = 1
weights = ot.unif(K)
Y_list = []
b_list = [ot.unif(n)] * K
cost_list = [ot.dist] * K
for _ in range(K):
    Y_list.append(np.random.randn(n, d))

X_init = np.random.randn(n, d)

# Fixed point
t0 = time()
X_fp, log_FP = solve_OT_barycenter_fixed_point(
    X_init, Y_list, b_list, cost_list, lambda y: B(y, weights),
    max_its=fp_its,
    stop_threshold=stop_threshold, log=True)
dt_FP = time() - t0
a_FP = ot.unif(n)
V_FP = V(X_fp, a_FP, Y_list, b_list, weights)
print(f'FP finished in {dt_FP:.5f}s, V={V_FP:.5f}')

# torch versions on cpu for GD
Y_list_torch = TT(Y_list, device=device)
b_list_torch = TT(b_list, device=device)
weights_torch = TT(weights, device=device)

# Gradient Descent
t0 = time()
X_gd, a_gd, log_GD = solve_OT_barycenter_GD(
    Y_list_torch, b_list_torch, weights_torch, cost_list, n, d, eta_init=gd_eta,
    its=gd_its, stop_threshold=stop_threshold, gamma=gd_gamma,
    a_unif=gd_a_unif, log=True)
dt_GD = time() - t0
V_GD = V(TN(X_gd), TN(a_gd), Y_list, b_list, weights)
print(f'GD finished in {dt_GD:.5f}s, V={V_GD:.5f}')


# %%

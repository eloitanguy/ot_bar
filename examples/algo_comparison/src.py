# %%
import numpy as np
import torch
from ot_bar.solvers import solve_w2_barycentre_multi_marginal, solve_OT_barycenter_fixed_point
from ot_bar.utils import plot_runs
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
    'stop_threshold': 1e-8,
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

fig, axs = plt.subplots(2, 1, figsize=(3, 6))
plot_runs(V_ratios, x=n_list, ax=axs[0],
          curve_labels=curve_labels, title='V / V MM', x_label='n', x_scale_log=False, y_scale_log=False, legend_loc='lower right',
          curve_colours=colours)
plot_runs(dt_ratios, x=n_list, ax=axs[1],
          curve_labels=curve_labels, title='Time / Time MM', x_label='n', x_scale_log=False, y_scale_log=True, curve_colours=colours)
plt.suptitle('FP vs MM different n',
             y=.95, fontsize=14)
plt.subplots_adjust(hspace=0.4)
plt.savefig(xp_name + '.pdf', bbox_inches='tight')
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
    'stop_threshold': 1e-8,
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

fig, axs = plt.subplots(2, 1, figsize=(3, 6))
plot_runs(V_ratios, x=d_list, ax=axs[0],
          curve_labels=curve_labels, title='V / V MM', x_label='d', x_scale_log=False, y_scale_log=False, curve_colours=colours)
plot_runs(dt_ratios, x=d_list, ax=axs[1],
          curve_labels=curve_labels, title='Time / Time MM', x_label='d', x_scale_log=False, y_scale_log=True, curve_colours=colours)
plt.suptitle('FP vs MM different d',
             y=.95, fontsize=14)
plt.subplots_adjust(hspace=0.4)
plt.savefig(xp_name + '.pdf', bbox_inches='tight')
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
    'stop_threshold': 1e-8,
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

fig, axs = plt.subplots(2, 1, figsize=(3, 6))
plot_runs(V_ratios, x=K_list, ax=axs[0],
          curve_labels=curve_labels, title='V / V MM', x_label='K', x_scale_log=False, y_scale_log=False, curve_colours=colours)
plot_runs(dt_ratios, x=K_list, ax=axs[1],
          curve_labels=curve_labels, title='Time / Time MM', x_label='K', x_scale_log=False, y_scale_log=True, curve_colours=colours,
          legend_loc='lower left')
plt.suptitle('FP vs MM different K',
             y=.95, fontsize=14)
plt.subplots_adjust(hspace=0.4)
plt.savefig(xp_name + '.pdf', bbox_inches='tight')
plt.show()

# %% plot results
V_results_n, dt_results_n = np.load(
    'multiple_n_d10_K3_results.npy', allow_pickle=True)
V_results_d, dt_results_d = np.load(
    'multiple_d_n30_K3_results.npy', allow_pickle=True)
V_results_K, dt_results_K = np.load(
    'multiple_K_n10_d10_results.npy', allow_pickle=True)

V_results_list = [V_results_n, V_results_d, V_results_K]
V_results_list = [V[(0, 1, -1), :] for V in V_results_list]
V_ratios_list = [x[:-1] / x[-1][None, :, :] for x in V_results_list]

dt_results_list = [dt_results_n, dt_results_d, dt_results_K]
dt_results_list = [dt[(0, 1, -1), :] for dt in dt_results_list]
dt_ratios_list = [x[:-1] / x[-1][None, :, :] for x in dt_results_list]

param_names = ['n', 'd', 'K']
param_values = [n_list, d_list, K_list]
curve_labels = [f'FP {fp_its} its' for fp_its in fp_its_list[:2]]

fig, axs = plt.subplots(2, 3, figsize=(12, 6))
for xp_idx in range(3):
    labels = curve_labels if xp_idx == 2 else None
    plot_runs(V_ratios_list[xp_idx], x=param_values[xp_idx], ax=axs[0, xp_idx],
              curve_labels=labels, title='V / V MM',
              x_label=param_names[xp_idx], x_scale_log=False,
              y_scale_log=False, curve_colours=colours)
    plot_runs(dt_ratios_list[xp_idx], x=param_values[xp_idx], ax=axs[1, xp_idx],
              curve_labels=None, title='Time / Time MM',
              x_label=param_names[xp_idx], x_scale_log=False, y_scale_log=True,
              curve_colours=colours, legend_loc='lower left')

plt.subplots_adjust(hspace=0.4)
plt.subplots_adjust(wspace=0.4)
plt.suptitle('Ratios of energy and time between FP and MM, varying $n, d, K$', fontsize=16)
plt.savefig('fp_vs_mm.pdf', bbox_inches='tight')
plt.show()

# %% Loops with n_FP = K(n-1) + 1 for multiple n and d=10, K=3
seed = 0
np.random.seed(seed)
torch.manual_seed(seed)
d = 10
K = 3
cost_list = [ot.dist] * K
n_list = [10, 30, 50, 70, 100]
fp_its_list = [1, 5, 10, 50]
n_samples = 10
xp_name = f'N_multiple_n_d{d}_K{K}'
results_file = xp_name + '_results.npy'
params_file = xp_name + '_params.json'

xp_params = {
    'seed': seed,
    'd': d,
    'K': K,
    'n_list': n_list,
    'n_samples': n_samples,
    'stop_threshold': 1e-8,
    'fp_its_list': fp_its_list,
    'gd_its': 1000,
    'gd_eta': 10,
    'gd_gamma': 1,
    'gd_a_unif': True,
    'mm_eps': 1e-5,
    'cost': 'L2'
}


# define the experiments
def run_N_multiple_n_xp():
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
            N = K * (n - 1) + 1
            X_init = np.random.randn(N, d)
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
                a_fp = ot.unif(N)
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
        run_N_multiple_n_xp()

if not os.path.exists(results_file) or not os.path.exists(params_file):
    print(f'No results found, running {xp_name}...')
    run_N_multiple_n_xp()

# load results
V_results, dt_results = np.load(results_file, allow_pickle=True)

# plot results
curve_labels = [f'FP {fp_its} its' for fp_its in fp_its_list[:3]]
V_ratios = V_results[:-1] / V_results[-1][None, :, :]
dt_ratios = dt_results[:-1] / dt_results[-1][None, :, :]

fig, axs = plt.subplots(2, 1, figsize=(3, 6))
plot_runs(V_ratios, x=n_list, ax=axs[0],
          curve_labels=curve_labels, title='V / V MM', x_label='n', x_scale_log=False, y_scale_log=False, legend_loc='lower right',
          curve_colours=colours)
plot_runs(dt_ratios, x=n_list, ax=axs[1],
          curve_labels=curve_labels, title='Time / Time MM', x_label='n', x_scale_log=False, y_scale_log=True, curve_colours=colours)
plt.suptitle('N FP vs MM different n',
             y=.95, fontsize=14)
plt.subplots_adjust(hspace=0.4)
plt.savefig(xp_name + '.pdf', bbox_inches='tight')
plt.show()

# %% Loops for multiple d and n=30, K=3
seed = 0
np.random.seed(seed)
torch.manual_seed(seed)
n = 30
K = 3
N = K * (n - 1) + 1
cost_list = [ot.dist] * K
d_list = [10, 30, 50, 70, 100]
fp_its_list = [1, 5, 10, 50]
n_samples = 10
xp_name = f'N_multiple_d_n{n}_K{K}'
results_file = xp_name + '_results.npy'
params_file = xp_name + '_params.json'

xp_params = {
    'seed': seed,
    'n': n,
    'N': N,
    'K': K,
    'd_list': d_list,
    'n_samples': n_samples,
    'stop_threshold': 1e-8,
    'fp_its_list': fp_its_list,
    'gd_its': 1000,
    'gd_eta': 10,
    'gd_gamma': 1,
    'gd_a_unif': True,
    'mm_eps': 1e-5,
    'cost': 'L2'
}


# define the experiments
def run_N_multiple_d_xp():
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
            X_init = np.random.randn(N, d)
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
                a_fp = ot.unif(N)
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
        run_N_multiple_d_xp()

if not os.path.exists(results_file) or not os.path.exists(params_file):
    print(f'No results found, running {xp_name}...')
    run_N_multiple_d_xp()

# load results
V_results, dt_results = np.load(results_file, allow_pickle=True)

# plot results
curve_labels = [f'FP {fp_its} its' for fp_its in fp_its_list[:3]]
V_ratios = V_results[:-1] / V_results[-1][None, :, :]
dt_ratios = dt_results[:-1] / dt_results[-1][None, :, :]

fig, axs = plt.subplots(2, 1, figsize=(3, 6))
plot_runs(V_ratios, x=d_list, ax=axs[0],
          curve_labels=curve_labels, title='V / V MM', x_label='d', x_scale_log=False, y_scale_log=False, curve_colours=colours)
plot_runs(dt_ratios, x=d_list, ax=axs[1],
          curve_labels=curve_labels, title='Time / Time MM', x_label='d', x_scale_log=False, y_scale_log=True, curve_colours=colours)
plt.suptitle('N FP vs MM different d',
             y=.95, fontsize=14)
plt.subplots_adjust(hspace=0.4)
plt.savefig(xp_name + '.pdf', bbox_inches='tight')
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
xp_name = f'N_multiple_K_n{n}_d{d}'
results_file = xp_name + '_results.npy'
params_file = xp_name + '_params.json'

xp_params = {
    'seed': seed,
    'n': n,
    'd': d,
    'K_list': K_list,
    'n_samples': n_samples,
    'stop_threshold': 1e-8,
    'fp_its_list': fp_its_list,
    'gd_its': 1000,
    'gd_eta': 10,
    'gd_gamma': 1,
    'gd_a_unif': True,
    'mm_eps': 1e-5,
    'cost': 'L2'
}


# define the experiments
def run_N_multiple_K_xp():
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
            N = K * (n - 1) + 1
            X_init = np.random.randn(N, d)
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
                a_fp = ot.unif(N)
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
        run_N_multiple_K_xp()

if not os.path.exists(results_file) or not os.path.exists(params_file):
    print(f'No results found, running {xp_name}...')
    run_N_multiple_K_xp()

# load results
V_results, dt_results = np.load(results_file, allow_pickle=True)

# plot results
curve_labels = [f'FP {fp_its} its' for fp_its in fp_its_list[:3]]
V_ratios = V_results[:-1] / V_results[-1][None, :, :]
dt_ratios = dt_results[:-1] / dt_results[-1][None, :, :]

fig, axs = plt.subplots(2, 1, figsize=(3, 6))
plot_runs(V_ratios, x=K_list, ax=axs[0],
          curve_labels=curve_labels, title='V / V MM', x_label='K', x_scale_log=False, y_scale_log=False, curve_colours=colours)
plot_runs(dt_ratios, x=K_list, ax=axs[1],
          curve_labels=curve_labels, title='Time / Time MM', x_label='K', x_scale_log=False, y_scale_log=True, curve_colours=colours,
          legend_loc='lower left')
plt.suptitle('N FP vs MM different K',
             y=.95, fontsize=14)
plt.subplots_adjust(hspace=0.4)
plt.savefig(xp_name + '.pdf', bbox_inches='tight')
plt.show()
# %%
V_results_N_n, dt_results_n = np.load('N_multiple_n_d10_K3_results.npy', allow_pickle=True)
V_results_N_d, dt_results_d = np.load('N_multiple_d_n30_K3_results.npy', allow_pickle=True)
V_results_N_K, dt_results_K = np.load('N_multiple_K_n10_d10_results.npy', allow_pickle=True)

V_results_list = [V_results_N_n, V_results_N_d, V_results_N_K]
V_ratios_list = [x[:-1] / x[-1][None, :, :] for x in V_results_list]

dt_results_list = [dt_results_n, dt_results_d, dt_results_K]
dt_ratios_list = [x[:-1] / x[-1][None, :, :] for x in dt_results_list]

param_names = ['n', 'd', 'K']
param_values = [n_list, d_list, K_list]

fig, axs = plt.subplots(2, 3, figsize=(12, 6))
for xp_idx in range(3):
    labels = curve_labels if xp_idx == 2 else None
    plot_runs(V_ratios_list[xp_idx], x=param_values[xp_idx], ax=axs[0, xp_idx],
              curve_labels=labels, title='V / V MM',
              x_label=param_names[xp_idx], x_scale_log=False,
              y_scale_log=False, curve_colours=colours)
    plot_runs(dt_ratios_list[xp_idx], x=param_values[xp_idx], ax=axs[1, xp_idx],
              curve_labels=None, title='Time / Time MM',
              x_label=param_names[xp_idx], x_scale_log=False, y_scale_log=True,
              curve_colours=colours, legend_loc='lower left')

plt.subplots_adjust(hspace=0.4)
plt.subplots_adjust(wspace=0.4)
plt.suptitle('Ratios of energy and time between FPH and MM, varying $n, d, K$ for $N = (n-1)K + 1$', fontsize=16)
plt.savefig('N_fp_vs_mm.pdf', bbox_inches='tight')
plt.show()

# %% Loops of G for multiple n and d=10, K=3
seed = 0
np.random.seed(seed)
torch.manual_seed(seed)
d = 10
K = 3
cost_list = [ot.dist] * K
n_list = [10, 30, 50, 70, 100]
fp_its_list = [1, 5, 10, 50]
n_samples = 10
xp_name = f'G_multiple_n_d{d}_K{K}'
results_file = xp_name + '_results.npy'
params_file = xp_name + '_params.json'

xp_params = {
    'seed': seed,
    'd': d,
    'K': K,
    'n_list': n_list,
    'n_samples': n_samples,
    'stop_threshold': 1e-8,
    'fp_its_list': fp_its_list,
    'gd_its': 1000,
    'gd_eta': 10,
    'gd_gamma': 1,
    'gd_a_unif': True,
    'mm_eps': 1e-5,
    'cost': 'L2'
}


# define the experiments
def run_G_multiple_n_xp():
    # idx 0 ... len(fp_its_list) - 1 : fp its, idx -1 : mm
    V_results = np.zeros((1 + len(fp_its_list), len(n_list), n_samples))
    dt_results = np.zeros((1 + len(fp_its_list), len(n_list), n_samples))
    iterator = tqdm(n_list)

    for n_idx, n in enumerate(iterator):
        for i in range(n_samples):
            iterator.set_postfix_str(
                f'n={n} [{n_idx + 1}/{len(n_list)}] sample {i + 1}/{n_samples} MM')
            Y_list = []
            nk_list = list(np.linspace(n // 2, 2 * n, K, dtype=int))
            b_list = [ot.unif(nk) for nk in nk_list]
            for nk in nk_list:
                Y_list.append(np.random.randn(nk, d))
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
                X_fp, a_fp = solve_OT_barycenter_fixed_point(
                    X_init, Y_list, b_list, cost_list, lambda y: B(y, weights),
                    max_its=fp_its, stop_threshold=xp_params['stop_threshold'],
                    method='true_fixed_point')
                dt_results[fp_idx, n_idx, i] = time() - t0
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
        run_G_multiple_n_xp()

if not os.path.exists(results_file) or not os.path.exists(params_file):
    print(f'No results found, running {xp_name}...')
    run_G_multiple_n_xp()

# load results
V_results, dt_results = np.load(results_file, allow_pickle=True)

# plot results
curve_labels = [f'FP {fp_its} its' for fp_its in fp_its_list[:3]]
V_ratios = V_results[:-1] / V_results[-1][None, :, :]
dt_ratios = dt_results[:-1] / dt_results[-1][None, :, :]

fig, axs = plt.subplots(2, 1, figsize=(3, 6))
plot_runs(V_ratios, x=n_list, ax=axs[0],
          curve_labels=curve_labels, title='V / V MM', x_label='n', x_scale_log=False, y_scale_log=False, legend_loc='lower right',
          curve_colours=colours)
plot_runs(dt_ratios, x=n_list, ax=axs[1],
          curve_labels=curve_labels, title='Time / Time MM', x_label='n', x_scale_log=False, y_scale_log=True, curve_colours=colours)
plt.suptitle('FPG vs MM different n',
             y=.95, fontsize=14)
plt.subplots_adjust(hspace=0.4)
plt.savefig(xp_name + '.pdf', bbox_inches='tight')
plt.show()

# %% Loops of G for multiple d and n=30, K=3
seed = 0
np.random.seed(seed)
torch.manual_seed(seed)
n = 30
K = 3
cost_list = [ot.dist] * K
d_list = [10, 30, 50, 70, 100]
fp_its_list = [1, 5, 10, 50]
n_samples = 10
xp_name = f'G_multiple_d_n{n}_K{K}'
results_file = xp_name + '_results.npy'
params_file = xp_name + '_params.json'

xp_params = {
    'seed': seed,
    'n': n,
    'K': K,
    'd_list': d_list,
    'n_samples': n_samples,
    'stop_threshold': 1e-8,
    'fp_its_list': fp_its_list,
    'gd_its': 1000,
    'gd_eta': 10,
    'gd_gamma': 1,
    'gd_a_unif': True,
    'mm_eps': 1e-5,
    'cost': 'L2'
}


# define the experiments
def run_G_multiple_d_xp():
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
            nk_list = list(np.linspace(n // 2, 2 * n, K, dtype=int))
            b_list = [ot.unif(nk) for nk in nk_list]
            for nk in nk_list:
                Y_list.append(np.random.randn(nk, d))
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
                X_fp, a_fp = solve_OT_barycenter_fixed_point(
                    X_init, Y_list, b_list, cost_list, lambda y: B(y, weights),
                    max_its=fp_its, stop_threshold=xp_params['stop_threshold'],
                    method='true_fixed_point')
                dt_results[fp_idx, d_idx, i] = time() - t0
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
        run_G_multiple_d_xp()

if not os.path.exists(results_file) or not os.path.exists(params_file):
    print(f'No results found, running {xp_name}...')
    run_G_multiple_d_xp()

# load results
V_results, dt_results = np.load(results_file, allow_pickle=True)

# plot results
curve_labels = [f'FP {fp_its} its' for fp_its in fp_its_list[:3]]
V_ratios = V_results[:-1] / V_results[-1][None, :, :]
dt_ratios = dt_results[:-1] / dt_results[-1][None, :, :]

fig, axs = plt.subplots(2, 1, figsize=(3, 6))
plot_runs(V_ratios, x=d_list, ax=axs[0],
          curve_labels=curve_labels, title='V / V MM', x_label='d', x_scale_log=False, y_scale_log=False, curve_colours=colours)
plot_runs(dt_ratios, x=d_list, ax=axs[1],
          curve_labels=curve_labels, title='Time / Time MM', x_label='d', x_scale_log=False, y_scale_log=True, curve_colours=colours)
plt.suptitle('FPG vs MM different d',
             y=.95, fontsize=14)
plt.subplots_adjust(hspace=0.4)
plt.savefig(xp_name + '.pdf', bbox_inches='tight')
plt.show()

# %% Loops of G for multiple K and n=10, d=10
seed = 1
np.random.seed(seed)
torch.manual_seed(seed)
n = 10
d = 10
K_list = [2, 3, 4, 5, 6]
fp_its_list = [1, 5, 10, 50]
n_samples = 10
xp_name = f'G_multiple_K_n{n}_d{d}'
results_file = xp_name + '_results.npy'
params_file = xp_name + '_params.json'

xp_params = {
    'seed': seed,
    'n': n,
    'd': d,
    'K_list': K_list,
    'n_samples': n_samples,
    'stop_threshold': 1e-8,
    'fp_its_list': fp_its_list,
    'gd_its': 1000,
    'gd_eta': 10,
    'gd_gamma': 1,
    'gd_a_unif': True,
    'mm_eps': 1e-5,
    'cost': 'L2'
}


# define the experiments
def run_G_multiple_K_xp():
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
            nk_list = list(np.linspace(n // 2, 2 * n, K, dtype=int))
            b_list = [ot.unif(nk) for nk in nk_list]
            for nk in nk_list:
                Y_list.append(np.random.randn(nk, d))
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
                X_fp, a_fp = solve_OT_barycenter_fixed_point(
                    X_init, Y_list, b_list, cost_list, lambda y: B(y, weights),
                    max_its=fp_its, stop_threshold=xp_params['stop_threshold'],
                    method='true_fixed_point')
                dt_results[fp_idx, K_idx, i] = time() - t0
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
        run_G_multiple_K_xp()

if not os.path.exists(results_file) or not os.path.exists(params_file):
    print(f'No results found, running {xp_name}...')
    run_G_multiple_K_xp()

# load results
V_results, dt_results = np.load(results_file, allow_pickle=True)

# plot results
curve_labels = [f'FP {fp_its} its' for fp_its in fp_its_list[:3]]
V_ratios = V_results[:-1] / V_results[-1][None, :, :]
dt_ratios = dt_results[:-1] / dt_results[-1][None, :, :]

fig, axs = plt.subplots(2, 1, figsize=(3, 6))
plot_runs(V_ratios, x=K_list, ax=axs[0],
          curve_labels=curve_labels, title='V / V MM', x_label='K', x_scale_log=False, y_scale_log=False, curve_colours=colours)
plot_runs(dt_ratios, x=K_list, ax=axs[1],
          curve_labels=curve_labels, title='Time / Time MM', x_label='K', x_scale_log=False, y_scale_log=True, curve_colours=colours,
          legend_loc='lower left')
plt.suptitle('FPG vs MM different K',
             y=.95, fontsize=14)
plt.subplots_adjust(hspace=0.4)
plt.savefig(xp_name + '.pdf', bbox_inches='tight')
plt.show()
# %%
V_results_N_n, dt_results_n = np.load(
    'G_multiple_n_d10_K3_results.npy', allow_pickle=True)
V_results_N_d, dt_results_d = np.load(
    'G_multiple_d_n30_K3_results.npy', allow_pickle=True)
V_results_N_K, dt_results_K = np.load(
    'G_multiple_K_n10_d10_results.npy', allow_pickle=True)

V_results_list = [V_results_N_n, V_results_N_d, V_results_N_K]
V_results_list = [V[(0, 1, -1), :] for V in V_results_list]
V_ratios_list = [x[:-1] / x[-1][None, :, :] for x in V_results_list]

dt_results_list = [dt_results_n, dt_results_d, dt_results_K]
dt_results_list = [dt[(0, 1, -1), :] for dt in dt_results_list]
dt_ratios_list = [x[:-1] / x[-1][None, :, :] for x in dt_results_list]

param_names = ['n', 'd', 'K']
param_values = [n_list, d_list, K_list]

fig, axs = plt.subplots(2, 3, figsize=(12, 6))
for xp_idx in range(3):
    labels = curve_labels if xp_idx == 2 else None
    plot_runs(V_ratios_list[xp_idx], x=param_values[xp_idx], ax=axs[0, xp_idx],
              curve_labels=labels, title='V / V MM',
              x_label=param_names[xp_idx], x_scale_log=False,
              y_scale_log=False, curve_colours=colours)
    plot_runs(dt_ratios_list[xp_idx], x=param_values[xp_idx], ax=axs[1, xp_idx],
              curve_labels=None, title='Time / Time MM',
              x_label=param_names[xp_idx], x_scale_log=False, y_scale_log=True,
              curve_colours=colours, legend_loc='lower left')

plt.subplots_adjust(hspace=0.4)
plt.subplots_adjust(wspace=0.4)
plt.suptitle('Ratios of energy and time between FPG and MM, varying $n, d, K$ for G and $n_1 = \\frac{n}{2} ... n_K = 2n$', fontsize=16)
plt.savefig('G_fp_vs_mm.pdf', bbox_inches='tight')
plt.show()

# %%

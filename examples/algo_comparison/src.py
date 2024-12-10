# %%
import numpy as np
import torch
from ot_bar.solvers import solve_w2_barycentre_multi_marginal, solve_OT_barycenter_fixed_point, solve_NLGWB_GD
from ot_bar.utils import TT, TN
from time import time
import matplotlib.pyplot as plt
import ot  # type: ignore


device = 'cpu'
stop_threshold = 1e-5

# %% Loops for multiple n and d=10, K=3
np.random.seed(42)
torch.manual_seed(42)
d = 10
K = 3
n_list = [10, 50, 100, 500]
n_samples = 10
times_dict = {}
bar_fixed_dict = {}
bar_mm_dict = {}
bar_gd_dict = {}

for n in n_list:
    bar_fixed_dict[n] = []
    bar_mm_dict[n] = []
    bar_gd_dict[n] = []
    b_list = [ot.unif(n)] * K
    for i in range(n_samples):
        Y_list = []
        for k in range(K):
            Y_list.append(np.random.randn(n, d))
        

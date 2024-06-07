import numpy as np
import torch
from gwbnl import *
from time import time
import matplotlib.pyplot as plt


np.random.seed(42)
# number of points of measure 1 and measure 2
K1, K2 = 400, 600

# dimensions of: measure 1, measure 2 and the barycentre
d1, d2, d = 2, 2, 2  
L = 500  # number of points of the optimised barycentre

# weights of the two measures
a_list = TT([ot.unif(K1), ot.unif(K2)])

# weights for the barycentre
weights = TT(ot.unif(2))

# map 1: R^2 -> R^2
P1 = lambda X: torch.sin(10 * X)  # element-wise sine

# maps 2: R^2 -> R^2
A2 = TT(torch.tensor([[2, .5], [0, 3]]))
def P2(X: torch.tensor):
    # assumes that X is of shape (batch, d2)
    # applies a quadratic map R^2 -> R^2 to X
    return (X @ A2) * X

P_list = [P1, P2]

# positions of the two measures: for a simple optimisation problem, we take
# X1 as the image of a [0, 1]^2 uniform measure by the map P1, 
# and X2 as the image of a [0, 1]^2 uniform measure by the map P2,
# the target Y should look uniform [0, 1]^2. 
X_list = [P1(TT(np.random.rand(K1, d1))), P2(TT(np.random.rand(K2, d2)))]

# optimiser parameters
learning_rate = 1  # initial learning rate
its = 500  # Gradient Descent iterations
stop_threshold = 1e-10  # stops if |loss_{t+1} - loss_{t}| < this
gamma = 0.995 # learning rate at step t is initial learning rate * gamma^t

t0 = time()
Y, b, loss_list, exit_status = solve_NLGWB_GD(X_list, a_list, weights, 
                                                P_list, L, d, return_exit_status=True, eta_init=learning_rate,
                                                its=its,stop_threshold=stop_threshold,
                                                gamma=gamma)
dt = time() - t0
print(f"Finished in {dt:.2f}s, exit status: {exit_status}, final loss: {loss_list[-1]:.2f}")

markers = ['o', 's']
alpha = .5

fig, ax = plt.subplots(1, 3)
for i in range(2):
    ax[i].scatter(TN(X_list[i][:, 0]), TN(X_list[i][:, 1]), 
                        label=f'X{i}', marker=markers[i], c='dodgerblue', alpha=alpha)
    PiY = P_list[i](Y)
    ax[i].scatter(TN(PiY[:, 0]), TN(PiY[:, 1]), label=f'P{i + 1}(Y)', 
                    marker='*', c='forestgreen', alpha=alpha)
    ax[i].legend()
ax[2].scatter(TN(Y[:, 0]), TN(Y[:, 1]), label='Y', marker='*', 
                c='forestgreen', alpha=alpha)
ax[2].legend()

plt.savefig("example_gwb_2d_non_linear.pdf", format="pdf", 
            bbox_inches="tight")
plt.show()
plt.clf()

plt.plot(loss_list)
plt.xlabel('iterations')
plt.ylabel('loss (logscale)')
plt.yscale('log')
plt.savefig("example_gwb_2d_non_linear_loss.pdf", format="pdf", 
            bbox_inches="tight")
plt.show()

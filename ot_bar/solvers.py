import ot  # type: ignore
import torch
from torch.optim import SGD
from torch.optim.lr_scheduler import ExponentialLR
from tqdm.auto import trange
from ot.backend import get_backend  # type: ignore
import numpy as np
import cvxpy as cvx
from itertools import product


class StoppingCriterionReached(Exception):
    pass


def solve_NLGWB_GD(X_list, a_list, weights, P_list, L, d, b_unif=True,
                   its=300,
                   eta_init=10,
                   gamma=.98, pbar=True, return_exit_status=False, stop_threshold=1e-5):
    r"""
    Solves the Nonlinear Generalised Wasserstein Barycentre problem using GD

    Args:
        X_list: list of p (K_i, d_i) measure points
        a_list: list of p (K_i) measure weights
        weights: array of the p barycentre coefficients
        P_list: list of p pytorch auto-differentiable functions
        :math:`\\mathbb{R}^{d_i} \\longrightarrow \\mathbb{R}^d`
        L: number of barycentre points to optimise
        d: dimension of the barycentre points
        b_unif: boolean toggling uniform barycentre or optimised weights
        its: (S)GD iterations
        eta_init: initial GD learning rate
        gamma: GD learning rate decay factor: :math:`\\eta_{t+1} = \\gamma \\eta_t`
        pbar: whether to display a progress bar
        return_exit_status: if True will also return the algorithm's exit status
        stop_threshold: if the loss changes less than this threshold between two steps, stop

    Returns:
        Y: array of (L, d) barycentre
        points b: array (L) of barycentre weights
        loss_list: list of (its) loss values each iteration
        (optional) exit_status: status of the algorithm at the final iteration
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    p = len(X_list)
    opt_params = []  # torch parameters for torch.optim.SGD

    # Init barycentre positions Y (normal)
    Y = torch.randn((L, d), device=device, dtype=torch.double,
                    requires_grad=True)
    opt_params.append({'params': Y})

    # Init barycentre coefficients b
    if not b_unif:
        b = torch.rand(L, device=device, dtype=torch.double, requires_grad=True)
        b = b / b.sum()
        b.requires_grad_()
        opt_params.append({'params': b})
    else:
        b = torch.tensor(ot.unif(L), device=device, dtype=torch.double,
                         requires_grad=False)

    # Prepare GD loop
    loss_list = [1e10]  # placeholder first value for stopping criterion check
    iterator = trange(its) if pbar else range(its)
    exit_status = 'Unknown'

    opt = SGD(opt_params, lr=eta_init)
    sch = ExponentialLR(opt, gamma)

    # GD loop
    try:
        for _ in iterator:
            opt.zero_grad()
            loss = 0
            # compute loss
            for i in range(p):
                M = ot.dist(X_list[i], P_list[i](Y))
                loss += weights[i] * ot.emd2(a_list[i], b, M)

            loss.backward()
            opt.step()
            sch.step()
            loss_list.append(loss.item())

            # stationary criterion: decrease of less than the threshold
            if stop_threshold > loss_list[-2] - loss_list[-1] >= 0:
                exit_status = 'Local optimum'
                raise StoppingCriterionReached

            # Update progress bar info
            if pbar:
                iterator.set_description('loss={:.5e}'.format(loss.item()))

            # Apply constraint projections to weights b
            with torch.no_grad():
                if not b_unif:
                    b.data = ot.utils.proj_simplex(b.data)

        # Finished loop: raise exception anyway to factorise code
        exit_status = 'Max iterations reached'
        raise StoppingCriterionReached

    except StoppingCriterionReached:
        if return_exit_status:
            return Y, b, loss_list[1:], exit_status
        else:
            return Y, b, loss_list[1:]


def solve_OT_barycenter_fixed_point(X, Y_list, b_list, cost_list, B,
                                    max_its=300, pbar=False, log=False):
    r"""
    Solves the OT barycenter problem using the fixed point algorithm, iterating
    the function B on plans between the current barycentre and the measures.

    Args:
        X: (n, d) array of barycentre points
        Y_list: list of K (n_k, d_k) arrays
        b_list: list of K (n_k) arrays of weights
        cost_list: list of K cost functions R^d x R^d_k -> R_+
        B: function from R^d_1 x ... x R^d_K to R^d accepting K arrays of shape
        (n, d_K)
        max_its: maximum number of iterations
        pbar: whether to display a progress bar
        log: whether to return the list of iterations

    Returns:
        X: (n, d) array of barycentre points
    """
    nx = get_backend(X, Y_list[0])
    K = len(Y_list)
    iterator = trange(max_its) if pbar else range(max_its)
    n = X.shape[0]
    a = nx.from_numpy(ot.unif(n), type_as=X)
    X_list = [X]

    for _ in iterator:
        pi_list = [ot.emd(a, b_list[k], cost_list[k](X, Y_list[k])) for k in range(K)]
        Y_perm = []
        for k in range(K):
            Y_perm.append(n * pi_list[k] @ Y_list[k])
        X = B(Y_perm)
        if log:
            X_list.append(X)

    if log:
        return X, X_list
    return X


def multi_marginal_L2_cost_tensor(Y_list, weights):
    r"""
    Computes the m_1 x ... x m_K tensor of costs for the multi-marginal problem
    for the L2 cost.
    """
    K = len(Y_list)
    m_list = [Y_list[k].shape[0] for k in range(K)]
    M = np.zeros(m_list)
    for indices in product(*[range(m) for m in m_list]):
        # indices is (j1, ..., jK)
        # y_slice is a K x d matrix Y1[j1] ... YK[jK]
        y_slice = np.stack([Y_list[k][indices[k]] for k in range(K)], axis=0)
        mean = weights @ y_slice  # (1, d)
        norms = np.sum((mean - y_slice)**2, axis=1)  # (K,)
        M[indices] = np.sum(weights * norms)
    return M


def solve_MMOT(b_list, M):
    r"""
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


def solve_w2_barycentre_multi_marginal(Y_list, b_list, weights, eps=1e-5):
    r""""
    Computes the W2 barycentre of the given measures y (m_k, d) with weights w
    using the multi-marginal solver. The output will consider that there is mass
    on a point if the mass is greater than eps / (m_1 * ... * m_K).
    Expects numpy arrays.
    """
    M = multi_marginal_L2_cost_tensor(Y_list, weights)
    pi = solve_MMOT(b_list, M)
    m_list = [len(b) for b in b_list]
    K = len(m_list)
    d = Y_list[0].shape[1]

    # indices with mass
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


def gmm_barycentre_cost_tensor(means_list, covs_list, weights):
    """
    Computes the m_1 x ... x m_K tensor of costs for the Gaussian Mixture multi-marginal problem.
    """
    K = len(means_list)
    m_list = [means_list[k].shape[0] for k in range(K)]
    M = np.zeros(m_list)
    for indices in product(*[range(m) for m in m_list]):
        # indices is (j1, ..., jK)
        # means is a K x d matrix means1[j1] ... meansK[jK]
        # covs is a K x d x d tensor covs1[j1] ... covsK[jK]
        means_slice = np.stack([means_list[k][indices[k]] for k in range(K)],
                               axis=0)
        covs_slice = np.stack([covs_list[k][indices[k]] for k in range(K)],
                              axis=0)
        mean_bar, cov_bar = ot.gaussian.bures_wasserstein_barycenter(
            means_slice, covs_slice, weights)
        # cost at j1 ... jK is the weighted sum over kW2 distance between the
        # barycentre of the (N(m_l_jl, C_l_jl))_l and N(m_k_jk, C_k_jk)
        cost = 0
        for k in range(K):
            cost += weights[k] * ot.gaussian.bures_wasserstein_distance(
                mean_bar, means_slice[k], cov_bar, covs_slice[k])
        M[indices] = cost
    return M


def solve_gmm_barycentre_multi_marginal(means_list, covs_list, w_list, weights, 
                                        eps=1e-5):
    r"""
    Computes the Mixed-W2 barycentre of the given GMMs with weights w
    using the multi-marginal solver. The output will consider that there is mass
    on a point if the mass is greater than eps / (m_1 * ... * m_K).
    Expects numpy arrays.
    """
    M = gmm_barycentre_cost_tensor(means_list, covs_list, weights)
    pi = solve_MMOT(w_list, M)
    K = len(means_list)
    m_list = [means_list[k].shape[0] for k in range(K)]
    d = means_list[0].shape[1]

    # indices with mass
    indices = np.where(pi > eps / np.prod(m_list))
    n = len(indices[0])  # size of the support of the solution
    a = pi[indices]  # barycentre weights

    # barycentre support
    means = np.zeros((n, d))
    covs = np.zeros((n, d, d))
    for i, idx_tuple in enumerate(zip(*indices)):
        # means is a K x d matrix means1[j1] ... meansK[jK]
        # covs is a K x d x d tensor covs1[j1] ... covsK[jK]
        means_slice = np.stack([means_list[k][idx_tuple[k]] for k in range(K)],
                               axis=0)
        covs_slice = np.stack([covs_list[k][idx_tuple[k]] for k in range(K)],
                              axis=0)
        # w2 barycentre of the slices
        mean_bar, cov_bar = ot.gaussian.bures_wasserstein_barycenter(
            means_slice, covs_slice, weights)
        means[i], covs[i] = mean_bar, cov_bar
    a = a / a.sum()
    return means, covs, a

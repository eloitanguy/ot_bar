import ot  # type: ignore
import torch
from torch.optim import SGD
from torch.optim.lr_scheduler import ExponentialLR
from tqdm.auto import trange
from ot.backend import get_backend  # type: ignore
import numpy as np
import cvxpy as cvx
from itertools import product
from ot.gmm import gmm_ot_plan  # type: ignore
from ot.gaussian import bures_wasserstein_barycenter  # type: ignore
from .utils import to_int_array, clean_discrete_measure


class StoppingCriterionReached(Exception):
    pass


def solve_OT_barycenter_GD(Y_list, b_list, weights, cost_list, n, d,
                           a_unif=True, its=300, eta_init=10, gamma=.98, pbar=False, stop_threshold=1e-5, log=False,
                           device=None):
    r"""
    Solves the Optimal Transport Barycentre problem using Gradient Descent on
    the positions (and optionally weights).

    Parameters
    ----------
    Y_list : list of array-like
        List of p (K_i, d_i) measure points.
    b_list : list of array-like
        List of p (K_i) measure weights.
    weights : array-like
        Array of the p barycentre coefficients.
    cost_list : list of callable
        List of K pytorch auto-differentiable functions computing cost matrices
        R^{n x d} x R^{n_k x d_k} -> R_+^{n x n_k}.
    n : int
        Number of barycentre points to optimise.
    d : int
        Dimension of the barycentre points.
    a_unif : bool, optional
        Boolean toggling uniform barycentre or optimised weights (default is True).
    its : int, optional
        (S)GD iterations (default is 300).
    eta_init : float, optional
        Initial GD learning rate (default is 10).
    gamma : float, optional
        GD learning rate decay factor: :math:`\\eta_{t+1} = \\gamma \\eta_t` (default is 0.98).
    pbar : bool, optional
        Whether to display a progress bar (default is True).
    stop_threshold : float, optional
        If the iterations move less than this (relatively), terminate.
    log : bool, optional
        Whether to return the list of iterations and exit status.
    device : str, optional
        Device to use for the computation (default is 'cuda' if available, else 'cpu').

    Returns
    -------
    X : array-like
        Array of (n, d) barycentre points.
    a : array-like
        Array (n) of barycentre weights.
    loss_list : list
        List of (its) loss values each iteration.
    exit_status : str, optional
        Status of the algorithm at the final iteration (only if return_exit_status is True)."""

    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    opt_params = []  # torch parameters for the optimiser
    K = len(Y_list)

    # Init barycentre positions Y (normal)
    X = torch.randn((n, d), device=device, dtype=torch.double,
                    requires_grad=True)
    opt_params.append({'params': X})

    # Init barycentre coefficients b
    if not a_unif:
        a = torch.rand(n, device=device, dtype=torch.double, requires_grad=True)
        a = a / a.sum()
        a.requires_grad_()
        opt_params.append({'params': a})
    else:
        a = torch.tensor(ot.unif(n), device=device, dtype=torch.double,
                         requires_grad=False)

    # Prepare GD loop
    loss_list = []

    iterator = trange(its) if pbar else range(its)
    exit_status = 'Unknown'

    opt = SGD(opt_params, lr=eta_init)
    sch = ExponentialLR(opt, gamma)

    # GD loop
    try:
        for _ in iterator:
            X_prev = X.data.clone()
            opt.zero_grad()
            loss = 0
            # compute loss
            for k in range(K):
                M = cost_list[k](X, Y_list[k])
                loss += weights[k] * ot.emd2(a, b_list[k], M)

            loss.backward()
            opt.step()
            sch.step()
            if log:
                loss_list.append(loss.item())

            with torch.no_grad():
                # stationary criterion: move less than the threshold
                diff = ot.emd2(a, a, ot.dist(X.data, X_prev))
                current = torch.sum(X_prev**2)
                if diff / current < stop_threshold:
                    exit_status = 'Local optimum'
                    raise StoppingCriterionReached

            # Update progress bar info
            if pbar:
                iterator.set_description('loss={:.5e}'.format(loss.item()))

            # Apply constraint projections to weights b
            with torch.no_grad():
                if not a_unif:
                    a.data = ot.utils.proj_simplex(a.data)

        # Finished loop: raise exception anyway to factorise code
        exit_status = 'Max iterations reached'
        raise StoppingCriterionReached

    except StoppingCriterionReached:
        if log:
            log_dict = {'loss_list': loss_list[1:], 'exit_status': exit_status}
            return X, a, log_dict
        else:
            return X, a


def solve_OT_barycenter_fixed_point(X, Y_list, b_list, cost_list, B, a=None,
                                    max_its=5, method='L2_barycentric_proj',
                                    stop_threshold=1e-2, pbar=False, log=False,
                                    clean_measure=False):
    """
    Solves the OT barycenter problem using the fixed point algorithm, iterating
    the function B on plans between the current barycentre and the measures.

    The method 'L2_barycentric_proj' uses Euclidean barycentric projections,
    imposing the support size to remain the same as the initial X, and fixing
    weights to remain equal to `a`.

    The method 'true_fixed_point' uses the North West Corner multi-marginal
    gluing method, which allows the support size N to increase at each
    iteration, and the weights to change. The final support size is
    upper-bounded by n + T(sum_k n_k - K), where n is the size of the initial
    barycentre X, n_k is the size of the k-th marginal, and T is the number of
    iterations.

    Parameters
    ----------
    X : array-like
        Array of shape (n, d) representing barycentre points.
    Y_list : list of array-like
        List of K arrays, each of shape (m_k, d_k).
    b_list : list of array-like
        List of K arrays, each of shape (m_k) representing weights.
    cost_list : list of callable
        List of K cost functions R^(n, d) x R^(m_k, d_k) -> R_+^(n, m_k).
    B : callable
        Function from R^d_1 x ... x R^d_K to R^d accepting a list of K arrays of
        shape (n, d_K), computing the "ground barycentre".
    a : array-like, optional
        Array of shape (n,) representing weights of the barycentre. If None, it
        is initialised to uniform. For the 'L2_barycentric_proj' method, a
        remains constant, and for the 'true_fixed_point' method, it is only the
        initialisation.
    max_its : int, optional
        Maximum number of iterations (default is 5).
    method : str, optional
        Barycentre method: 'L2_barycentric_proj' (default) for Euclidean
        barycentric projection, or 'true_fixed_point' for iterates using the
        North West Corner multi-marginal gluing method.
    stop_threshold : float, optional
        If the iterations move less than this (relatively), terminate.
    pbar : bool, optional
        Whether to display a progress bar (default is False).
    log : bool, optional
        Whether to return the list of iterations (default is False).
    clean_measure : bool, optional
        For method=='true_fixed_point', whether to clean the discrete measure
        (X, a) at each iteration to remove duplicate points and sum their
        weights (default is False).

    Returns
    -------
    X : array-like
        Array of shape (n, d) representing barycentre points.
    a : array-like
        Array of shape (n,) representing barycentre weights. Only returned if
        method is 'true_fixed_point'.
    log_dict : dict, optional
        log containing the exit status and list of iterations if log is True.
    """
    nx = get_backend(X, Y_list[0])
    K = len(Y_list)
    iterator = trange(max_its) if pbar else range(max_its)
    n = X.shape[0]
    if a is None:
        a = nx.from_numpy(ot.unif(n), type_as=X)
    X_list = [nx.copy(X)] if log else []
    a_list = [nx.copy(a)] if log and method == 'true_fixed_point' else []
    exit_status = 'Unknown'
    assert method in ['L2_barycentric_proj', 'true_fixed_point'], \
        "Method must be 'L2_barycentric_proj' or 'true_fixed_point'"

    try:
        for _ in iterator:
            X_prev = nx.copy(X)
            a_prev = nx.copy(a) if method == 'true_fixed_point' else a

            pi_list = [
                ot.emd(a, b_list[k], cost_list[k](X, Y_list[k]))
                for k in range(K)]

            Y_perm = []  # list of K arrays of shape (N, d_k) to apply B to
            if method == 'L2_barycentric_proj':
                for k in range(K):  # L2 barycentric projection of pi_k
                    Y_perm.append((1 / a[:, None]) * pi_list[k] @ Y_list[k])
            elif method == 'true_fixed_point':
                # North West Corner gluing of pi_k
                J, w = NorthWestMMGluing(pi_list)
                # J is a (N, K) array of indices, w is a (N,) array of weights
                # Each Y_perm[k] is a (N, d_k) array of some points in Y_list[k]
                Y_perm = [Y_list[k][J[:, k]] for k in range(K)]
                a = w  # update barycentre weights

            X = B(Y_perm)

            if clean_measure and method == 'true_fixed_point':
                # clean the discrete measure (X, a) to remove duplicates
                X, a = clean_discrete_measure(X, a)

            if log:
                X_list.append(X)
                if method == 'true_fixed_point':
                    a_list.append(a)

            # stationary criterion: move less than the threshold
            diff = ot.emd2(a, a_prev, ot.dist(X, X_prev))
            current = nx.sum(X_prev**2) / X_prev.shape[0]
            if diff / current < stop_threshold:
                exit_status = 'Stationary Point'
                raise StoppingCriterionReached

        exit_status = 'Max iterations reached'
        raise StoppingCriterionReached

    except StoppingCriterionReached:
        if log:
            log_dict = {
                'X_list': X_list, 'exit_status': exit_status, 'a_list': a_list}
            if method == 'true_fixed_point':
                return X, a, log_dict
            else:
                return X, log_dict

        if method == 'true_fixed_point':
            return X, a
        else:
            return X


def multi_marginal_L2_cost_tensor(Y_list, weights):
    r"""
    Computes the m_1 x ... x m_K tensor of costs for the multi-marginal problem
    for the L2 cost.

    Parameters
    ----------
    Y_list : list of array-like
        List of K arrays, each of shape (m_k, d), representing the points in each marginal.
    weights : array-like
        Array of shape (K,), representing the weights for each marginal.

    Returns
    -------
    M : array-like
        A m_1 x ... x m_K tensor of costs for the multi-marginal problem.
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
    matrix M of shape (m_1, ..., m_K) and measure weights b_list (b_1, ..., b_K) with b_k in the m_k simplex.
    Solves the Multi-Marginal Optimal Transport (MMOT) problem.

    Parameters
    ----------
    b_list : list of array-like
        List of measure weights, where each element b_k is in the m_k simplex.
    M : array-like
        Cost matrix of shape (m_1, ..., m_K).

    Returns
    -------
    pi : array-like
        An optimal coupling pi of shape (m_1, ..., m_K).

    Notes
    -----
    This function solves the MMOT problem using convex optimization.
    The implementation is adapted from
    https://github.com/judelo/gmmot/blob/master/python/gmmot.py#L210.
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


def solve_w2_barycentre_multi_marginal(Y_list, b_list, weights, eps=1e-5,
                                       clean_measure=False):
    r"""
    Computes the W2 barycentre of the given measures Y_list with weights using
    the multi-marginal solver. The output will consider that there is mass on a
    point if the mass is greater than eps / (m_1 * ... * m_K).

    Parameters
    ----------
    Y_list : list of array-like
        List of K arrays, each of shape (m_k, d), representing the points in
        each marginal.
    b_list : list of array-like
        List of K arrays, each of shape (m_k) representing weights.
    weights : array-like
        Array of shape (K,), representing the weights for each marginal.
    eps : float, optional
        Threshold for considering mass on a point (default is 1e-5).
    clean_measure : bool, optional
        Whether to clean the discrete measure (X, a) to remove duplicates and
        sum their weights (default is False).

    Returns
    -------
    X : array-like
        Array of shape (n, d) representing barycentre points.
    a : array-like
        Array of shape (n,) representing barycentre weights.
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

    if clean_measure:
        X, a = clean_discrete_measure(X, a)
    return X, a


def gmm_barycentre_cost_tensor(means_list, covs_list, weights):
    """
    Computes the m_1 x ... x m_K tensor of costs for the Gaussian Mixture multi-marginal problem.

    Parameters
    ----------
    means_list : list of array-like
        List of K arrays, each of shape (m_k, d), representing the means in each marginal.
    covs_list : list of array-like
        List of K arrays, each of shape (m_k, d, d), representing the covariances in each marginal.
    weights : array-like
        Array of shape (K,), representing the weights for each marginal.

    Returns
    -------
    M : array-like
        A m_1 x ... x m_K tensor of costs for the multi-marginal problem.
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
        mean_bar, cov_bar = bures_wasserstein_barycenter(
            means_slice, covs_slice, weights)
        # cost at j1 ... jK is the weighted sum over kW2 distance between the
        # barycentre of the (N(m_l_jl, C_l_jl))_l and N(m_k_jk, C_k_jk)
        cost = 0
        for k in range(K):
            cost += weights[k] * ot.gaussian.bures_wasserstein_distance(
                mean_bar, means_slice[k], cov_bar, covs_slice[k])
        M[indices] = cost
    return M


def solve_gmm_barycentre_multi_marginal(means_list, covs_list, b_list, weights,
                                        eps=1e-5):
    """
    Computes the Mixed-W2 barycentre of the given GMMs with weights using the
    multi-marginal solver. The output will consider that there is mass on a
    point if the mass is greater than eps / (m_1 * ... * m_K).

    Parameters
    ----------
    means_list : list of array-like
        List of K arrays, each of shape (m_k, d), representing the means in each marginal.
    covs_list : list of array-like
        List of K arrays, each of shape (m_k, d, d), representing the covariances in each marginal.
    b_list : list of array-like
        List of K arrays, each of shape (m_k) representing weights.
    weights : array-like
        Array of shape (K,), representing the weights for each marginal.
    eps : float, optional
        Threshold for considering mass on a point (default is 1e-5).

    Returns
    -------
    means : array-like
        Array of shape (n, d) representing barycentre means.
    covs : array-like
        Array of shape (n, d, d) representing barycentre covariances.
    a : array-like
        Array of shape (n,) representing barycentre weights.
    """
    M = gmm_barycentre_cost_tensor(means_list, covs_list, weights)
    pi = solve_MMOT(b_list, M)
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
        # means_slice is a K x d matrix means1[j1] ... meansK[jK]
        # covs_slice is a K x d x d tensor covs1[j1] ... covsK[jK]
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


def solve_gmm_barycenter_fixed_point(means, covs,
                                     means_list, covs_list, b_list, weights,
                                     max_its=300, pbar=False, log=False,
                                     barycentric_proj_method='euclidean'):
    r"""
    Solves the GMM OT barycenter problem using the fixed point algorithm.

    Parameters
    ----------
    means : array-like
        Initial (n, d) GMM means.
    covs : array-like
        Initial (n, d, d) GMM covariances.
    means_list : list of array-like
        List of K (m_k, d) GMM means.
    covs_list : list of array-like
        List of K (m_k, d, d) GMM covariances.
    b_list : list of array-like
        List of K (m_k) arrays of weights.
    weights : array-like
        Array (K,) of the barycentre coefficients.
    max_its : int, optional
        Maximum number of iterations (default is 300).
    pbar : bool, optional
        Whether to display a progress bar (default is False).
    log : bool, optional
        Whether to return the list of iterations (default is False).
    barycentric_proj_method : str, optional
        Method to project the barycentre weights: 'euclidean' (default) or 'bures'.

    Returns
    -------
    means : array-like
        (n, d) barycentre GMM means.
    covs : array-like
        (n, d, d) barycentre GMM covariances.
    log_dict : dict, optional
        Dictionary containing the list of iterations if log is True.
    """
    nx = get_backend(means, covs[0], means_list[0], covs_list[0])
    K = len(means_list)
    n = means.shape[0]
    d = means.shape[1]
    iterator = trange(max_its) if pbar else range(max_its)
    means_its = [means.copy()]
    covs_its = [covs.copy()]
    a = nx.from_numpy(ot.unif(n), type_as=means)

    for _ in iterator:
        pi_list = [
            gmm_ot_plan(means, means_list[k], covs, covs_list[k], a, b_list[k]) for k in range(K)]

        means_selection, covs_selection = None, None
        # in the euclidean case, the selection of Gaussians from each K sources
        # comes from a  barycentric projection is a convex combination of the
        # selected means and  covariances, which can be computed without a
        # for loop on i
        if barycentric_proj_method == 'euclidean':
            means_selection = nx.zeros((n, K, d), type_as=means)
            covs_selection = nx.zeros((n, K, d, d), type_as=means)

            for k in range(K):
                means_selection[:, k, :] = n * pi_list[k] @ means_list[k]
                covs_selection[:, k, :, :] = nx.einsum(
                    'ij,jab->iab', pi_list[k], covs_list[k]) * n

        # each component i of the barycentre will be a Bures barycentre of the
        # selected components of the K GMMs. In the 'bures' barycentric
        # projection option, the selected components are also Bures barycentres.
        for i in range(n):
            # means_slice_i (K, d) is the selected means, each comes from a
            # Gaussian barycentre along the disintegration of pi_k at i
            # covs_slice_i (K, d, d) are the selected covariances
            means_selection_i = []
            covs_selection_i = []

            # use previous computation (convex combination)
            if barycentric_proj_method == 'euclidean':
                means_selection_i = means_selection[i]
                covs_selection_i = covs_selection[i]

            # compute Bures barycentre of the selected components
            elif barycentric_proj_method == 'bures':
                for k in range(K):
                    w = (1 / a[i]) * pi_list[k][i, :]
                    m, C = bures_wasserstein_barycenter(
                        means_list[k], covs_list[k], w)
                    means_selection_i.append(m)
                    covs_selection_i.append(C)

            else:
                raise ValueError('Unknown barycentric_proj_method')

            means[i], covs[i] = bures_wasserstein_barycenter(
                means_selection_i, covs_selection_i, weights)

        if log:
            means_its.append(means.copy())
            covs_its.append(covs.copy())

    if log:
        return means, covs, {'means_its': means_its, 'covs_its': covs_its}
    return means, covs


def NorthWestMMGluing(pi_list, precision=1e-15, log=False):
    """
    Glue the transport plans (pi_1, ..., pi_K) using the (multi-marginal)
    North-West Corner method. Instead of outputting the full K+1-multi-marginal
    transport plan gamma in Pi(a, b_1, ..., b_K) with marginals gamma_0k pi_k,
    this function provides an array J of shape (N, K) where each J[i] is of the
    form (J[i, 1], ..., J[i, K]) with each J[i, k] between 0 and n_k-1, and a
    weight vector w of size N, such that the K-marginal rho := gamma_1...K =
    sum_i w_i delta_{y_{1, J[i, 1]}, ..., y_{K, J[i, K]}}, or in terms of
    tensors, rho[j_1, ..., j_K] = 1(exists i : (j_1, ..., j_K) = J[i]) w_i. This
    representation is useful for its memory efficiency, as it avoids storing the
    full K+1-marginal transport plan.

    If log=True, the function computes the full K+1-marginal transport plan and
    stores it in log_dict['gamma].

    Parameters
    ----------
    pi_list : list of arrays (n, n_k)
        List of transport plans.
    precision : float, optional
        Precision for the North-West Corner method (considering that
        |x|<precision means x==0). Default is 1e-15.
    log : bool, optional
        If True, return a log dictionary (computationally expensive).

    Returns
    -------
    J : array (N, K)
        The indices (J[i, 1], ..., J[i, K]) of the K-plan rho.
    w : array (N,)
        The weights w_i of the K-plan rho.
    log_dict : dict, optional
        If log=True, a dictionary containing the full K+1-marginal transport
        plan under the key 'gamma'.
    """
    nx = get_backend(pi_list[0])
    a = nx.sum(pi_list[0], axis=1)  # common first marginal a in Delta_n
    nk_list = [pi.shape[1] for pi in pi_list]  # list of n_k
    K = len(pi_list)
    n = pi_list[0].shape[0]  # number of points in the first marginal
    gamma = None

    log_dict = {'precision': precision}
    if log:  # n x n_1 x ... x n_K tensor
        gamma = nx.zeros([n] + nk_list, type_as=pi_list[0])

    gamma_weights = {}  # dict of (j_1, ..., j_K) : weight
    P_list = [nx.copy(pi) for pi in pi_list]  # copy of the transport plans

    for i in range(n):
        # jjs[k] is the list of indices j in [0, n_k - 1] such that Pk[i, j] >0
        jjs = [nx.to_numpy(nx.where(P[i, :] > precision)[0]) for P in P_list]
        # list [0, ..., 0] of size K for use with jjs: current indices in jjs
        jj_idx = [0] * K
        u = a[i]  # mass at i, will decrease to 0 as we fill gamma[i, :]

        while u > precision:  # while there is mass to add to gamma[i, :]
            # current multi-index j_1 ... j_K
            jj = tuple(jjs[k][jj_idx[k]] for k in range(K))
            # min transport plan value: min_k pi_k[i, j_k]
            v = nx.min(nx.stack([P_list[k][i, jj[k]] for k in range(K)]))
            if log:  # assign mass v to gamma[i, j_1, ..., j_K]
                gamma[(i,) + jj] = v
            if jj in gamma_weights:
                gamma_weights[jj] += v
            else:
                gamma_weights[jj] = v
            u -= v  # at i, we u-v mass left to assign
            for k in range(K):  # update plan copies Pk
                P_list[k][i, jj[k]] -= v  # Pk[i, j_k] has v less mass left
                if P_list[k][i, jj[k]] == 0:
                    # move to next index in jjs[k] if Pk[i, j_k] is empty
                    jj_idx[k] += 1

    log_dict['gamma'] = gamma
    J = list(gamma_weights.keys())  # list of multi-indices (j_1, ..., j_K)
    J = to_int_array(nx.from_numpy(np.array(J), type_as=pi_list[0]))
    w = nx.stack(list(gamma_weights.values()))
    if log:
        return J, w, log_dict
    return J, w

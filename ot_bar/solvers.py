import ot  # type: ignore
import torch
from torch.optim import SGD
from torch.optim.lr_scheduler import ExponentialLR
from tqdm.auto import trange
from ot.backend import get_backend  # type: ignore


class StoppingCriterionReached(Exception):
    pass


def solve_NLGWB_GD(X_list, a_list, weights, P_list, L, d, b_unif=True,
                   its=300,
                   eta_init=10,
                   gamma=.98, pbar=True, return_exit_status=False, stop_threshold=1e-5):
    """
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


def solve_OT_barycenter_fixed_point(X, Y_list, cost_list, B,
                                    max_its=300, pbar=False):
    """
    Solves the OT barycenter problem using the fixed point algorithm, iterating
    the function B on plans between the current barycentre and the measures.

    Args:
        X: (n, d) array of barycentre points
        Y_list: list of K (n_k, d_k) arrays
        cost_list: list of K cost functions
        B: function from R^d_1 x ... x R^d_K to R^d accepting K arrays of shape
        (n, d_K)
        max_its: maximum number of iterations
        pbar: whether to display a progress bar

    Returns:
        X: (n, d) array of barycentre points
    """
    nx = get_backend(X, Y_list[0])
    K = len(Y_list)
    iterator = trange(max_its) if pbar else range(max_its)
    n = X.shape[0]
    a = nx.from_numpy(ot.unif(n))

    for _ in iterator:
        pi_list = [ot.emd(a, a, cost_list[k](X, Y_list[k])) for k in range(K)]
        Y_perm = []
        for k in range(K):
            Y_perm.append(n * pi_list[k] @ Y_list[k])
        X = B(Y_perm)

    return X

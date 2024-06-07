import numpy as np
import ot
import torch
from torch.optim import SGD
from torch.optim.lr_scheduler import ExponentialLR
from tqdm.auto import trange


class StoppingCriterionReached(Exception):
    pass


def TN(x):
    """
    Returns a numpy version of the array or list of arrays given as input

    Args:
        x: torch tensor or list thereof
    """
    if isinstance(x, list) or isinstance(x, tuple):
        return [TN(o) for o in x]

    if torch.is_tensor(x):
        return x.detach().cpu().numpy()

    if isinstance(x, np.ndarray):
        return x

    raise TypeError('Expected a numpy array or a torch tensor')


def TT(x):
    """
    Returns a torch version (cuda if possible and dtype = double)
    of the array or list of arrays given as input

    Args:
        x: numpy tensor or list thereof
    """
    if isinstance(x, list) or isinstance(x, tuple):
        return [TT(o) for o in x]

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if isinstance(x, np.ndarray):
        return torch.tensor(x, dtype=torch.double, device=device)

    if torch.is_tensor(x):
        if device not in str(x.device):  # check if on the right device
            x = x.to(device)
        if x.dtype != torch.double:
            x = x.double()
        return x

    raise TypeError('Expected a numpy array or a torch tensor')


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
        :math:`\mathbb{R}^{d_i} \longrightarrow \mathbb{R}^d`
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

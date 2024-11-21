import torch
import numpy as np


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


def imageToGrid(image, constant_threshold=True):
    """
    Convert an image to an array (n_points,2) of points via thresholding
    """

    def condition(p):
        if constant_threshold:
            return p < 122.5
        return p > np.max(image) * 0.5

    x, y = image.shape

    points = []

    for i in range(x):
        for j in range(y):
            if condition(image[i, j]):
                points.append([1 - 1. * i / x, 1. * j / y])

    return np.array(points)

import numpy as np
import matplotlib.pyplot as plt
from os.path import abspath, dirname, join

seed = 42
rng = np.random.default_rng(seed)


# a: Modify function for nonreversal random walk
def grid_rw(N):
    """Generate N-step 2D random walk

    Parameters
    ----------
    N : int
        Length of random walk

    Returns
    -------
    arraylike
        2D array of the random walk
    """

    # Walk starts at 0
    x = 0
    y = 0

    # Will store the coordinate after each step.
    xs = []
    ys = []

    for step in np.arange(N):
        match step:
            case 0:
                x = x + 1
            case 1:
                x = x - 1
            case 2:
                y = y + 1
            case 3:
                y = y - 1

        xs.append(x)
        ys.append(y)

    return np.array([xs, ys])

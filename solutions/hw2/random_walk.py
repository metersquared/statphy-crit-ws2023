import numpy as np
import matplotlib.pyplot as plt
from os.path import abspath, dirname, join

seed = 42
rng = np.random.default_rng(seed)


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
    # Each integer represents a choice of up, down, left, right.
    steps = rng.integers(4, size=N)

    # Walk starts at 0
    x = 0
    y = 0

    # Will store the coordinate after each step.
    xs = []
    ys = []

    for step in steps:
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


# 4a: Modify function for nonreversal random walk


def opposite_dir(i):
    """Returns the opposite direction

    Parameters
    ----------
    i : int
        The direction index of a specific step

    Returns
    -------
    int
        The direction index of the opposite step
    """
    match i:
        case 0:
            return 1
        case 1:
            return 0
        case 2:
            return 3
        case 3:
            return 2


def next_coordinate(i: int, x, y):
    """Return the next coordinate following a step i.

    Parameters
    ----------
    i : int
        The direction index
    x : float/double whatevs
        x-coordinate
    y : float/double whatevs
        y-coordinate

    Returns
    -------
    tuple(x,y)
        x,y coordinate
    """
    match i:
        case 0:
            x = x + 1
        case 1:
            x = x - 1
        case 2:
            y = y + 1
        case 3:
            y = y - 1

    return x, y


def grid_rw_nonreversal(N):
    """Generate N-step 2D nonreversal random walk

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
    xs = np.zeros(N)
    ys = np.zeros(N)

    old_step = 4

    for i in np.arange(N):
        valid_step = False

        # Loop runs while a step is not valid, continously generating random steps
        while valid_step == False:
            step = rng.integers(4)

            # Nonreversal check: if old step is the opposite direction of current generated step, if yes, redo RNG
            if old_step == opposite_dir(step):
                continue

            x, y = next_coordinate(step, x, y)

            old_step = step

            valid_step = True

        xs[i] = x
        ys[i] = y

    return np.array([xs, ys])


# 4c : Implement self-avoiding walk
def grid_rw_selfavoiding(N):
    """Generate N-step 2D selfavoiding random walk

    Parameters
    ----------
    N : int
        Length of random walk

    Returns
    -------
    arraylike
        2D array of the random walk
    """

    trials = 0
    trial_success = False

    while trial_success == False:
        trials = trials + 1
        old_step = 4
        # Walk starts at 0
        x = 0
        y = 0
        # Will store the coordinate after each step.
        xs = np.zeros(N)
        ys = np.zeros(N)

        end_trial = False

        for i in np.arange(N):
            valid_step = False

            # Loop runs while a step is not valid, continously generating random steps
            while valid_step == False:
                step = rng.integers(4)

                # Nonreversal check: if old step is the opposite direction of current generated step, if yes, redo RNG
                if old_step == opposite_dir(step):
                    continue

                # Creates a new coordinate for next step
                x_new, y_new = next_coordinate(step, x, y)

                # Checks if new coordinate is already traversed
                coord_occupied = (x_new, y_new) in list(
                    map(tuple, np.array([xs[: i - 1], ys[: i - 1]]).T)
                )

                if coord_occupied:
                    nn_occupied = False
                    # Checks if there is traversable nearest neighbors, if no, then end trial and redo walk
                    for nn_step in np.arange(4):
                        x_nn, y_nn = next_coordinate(nn_step, x, y)
                        nn_occupied = (x_nn, y_nn) in list(
                            map(tuple, np.array([xs[: i - 1], ys[: i - 1]]).T)
                        )
                    if nn_occupied:
                        end_trial = True
                        break

                    continue

                x = x_new
                y = y_new

                old_step = step

                valid_step = True

            if end_trial:
                break

            xs[i] = x
            ys[i] = y

        if end_trial:
            continue

        trial_success = True

    return np.array([xs, ys]), trials


if __name__ == "__main__":
    # Test nonreversal walks

    plt.rcParams["text.usetex"] = True

    N = 50

    coords = grid_rw_nonreversal(N)

    fig1, axs1 = plt.subplots(sharey=True, tight_layout=True)
    fig1.suptitle("Nonreversal Random Walk")

    line = axs1.plot(coords[0], coords[1], color="black")
    axs1.axis("equal")
    axs1.set_box_aspect(1)

    axs1.set_ylabel("y")
    axs1.set_xlabel("x")

    path = join(dirname(abspath(__file__)), "hw04a_rw_nonreversal.pdf")

    fig1.savefig(path)

    # 4c 5 plots of self-avoiding random walks

    N = 50

    fig2, axs2 = plt.subplots(1, 5, figsize=(16, 4), sharey=True, tight_layout=True)
    fig2.suptitle("Selfavoiding Random Walk")

    for i in range(5):
        coords, trials = grid_rw_selfavoiding(N)
        axs2[i].plot(coords[0], coords[1], color="black")
        axs2[i].axis("equal")
        axs2[i].set_box_aspect(1)
        axs2[i].set_ylabel("trials={}".format(trials))
        axs2[i].set_xlabel("length={}".format(len(coords[0])))
        print(coords)

    path = join(dirname(abspath(__file__)), "hw04c_rw_selfavoiding.pdf")

    fig2.savefig(path)

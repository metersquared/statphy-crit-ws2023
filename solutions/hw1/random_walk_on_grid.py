import numpy as np
import matplotlib.pyplot as plt
from os.path import abspath, dirname, join

seed = 42
rng = np.random.default_rng(seed)


# a: Write a function that generates an N-step 2D random walk.
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


# b: Plot 3 random walks

plt.rcParams["text.usetex"] = True

N = 1000

fig1, axs1 = plt.subplots(1, 3, sharey=True, tight_layout=True)
fig1.suptitle("Some random walks")

for i in range(3):
    coords = grid_rw(N)
    axs1[i].plot(coords[0], coords[1], color="black")
    axs1[i].axis("equal")
    axs1[i].set_box_aspect(1)

axs1[0].set_ylabel("y")
axs1[1].set_xlabel("x")

path = join(dirname(abspath(__file__)), "hw02b_rw.pdf")

fig1.savefig(path)


# c: Perform 1000 random walk for 100 different N = 10,20,30...990,1000.

N_array = np.arange(
    10, 1010, 10
)  # Creates an array of N values from 10 to 1000 with step 10
MSQ_array = (
    []
)  # Will store the correspondong mean squared end-to-end distance of random walk
ERR_array = []

for N in N_array:  # Iterate over each N
    dist_SQ_array = []  # Stores the individual distance squared of the random walk

    for i in np.arange(100):
        coords = grid_rw(N)  # Generate a random walk
        dist_SQ_array.append(
            np.power(coords[0][-1], 2) + np.power(coords[1][-1], 2)
        )  # Calculate distance squared

    MSQ_array.append(np.mean(dist_SQ_array))  # Calculate mean an add to arrays of MSQ
    ERR_array.append(np.std(dist_SQ_array, ddof=1) / np.sqrt(np.size(dist_SQ_array)))

# Linear fit

coeff = np.polyfit(N_array, MSQ_array, 1)
p = np.poly1d(coeff)

# Plot the mean square end-to-end distance over N

fig2, ax2 = plt.subplots(sharey=True, tight_layout=True)
fig2.suptitle("Mean squared end-to-end distance of 2D random walks")

ax2.errorbar(N_array, MSQ_array, yerr=ERR_array, fmt="xk")
ax2.plot(N_array, p(N_array))

ax2.legend(
    ["Fit:${}+{}x$".format(coeff[0], coeff[1]), "Simulated walk"], loc="upper left"
)

path = join(dirname(abspath(__file__)), "hw02c_msqd_plot.pdf")

fig2.savefig(path)


# c: Perform 1000 random walk for 3 different N = 1000, 2000, 3000.

N_array = np.arange(1000, 4000, 1000)
bins = 25

fig4, axs4 = plt.subplots(3, 1, sharey=True, tight_layout=True)
fig5, axs5 = plt.subplots(3, 1, sharey=True, tight_layout=True)

fig4.suptitle("Histogram of end-to-end $x$ distances")
fig5.suptitle("Histogram of end-to-end $y$ distances")

for idx, N in enumerate(N_array):
    xs = []
    ys = []

    for i in np.arange(1000):
        coords = grid_rw(N)
        xs.append(coords[0][-1])
        ys.append(coords[1][-1])

    axs4[idx].hist(xs, bins=bins)
    axs5[idx].hist(ys, bins=bins)

    axs4[idx].set_xlim(-140, 140)
    axs5[idx].set_xlim(-140, 140)
    axs4[idx].set_ylim(0, 150)
    axs5[idx].set_ylim(0, 150)

    axs4[idx].set_ylabel("$N={}$".format(N))
    axs5[idx].set_ylabel("$N={}$".format(N))


axs4[-1].set_xlabel("$x(N)-x(0)$")
axs5[-1].set_xlabel("$y(N)-y(0)$")

path = join(dirname(abspath(__file__)), "hw02d_d_plotx.pdf")
fig4.savefig(path)

path = join(dirname(abspath(__file__)), "hw02d_d_ploty.pdf")
fig5.savefig(path)

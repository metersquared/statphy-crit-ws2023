import numpy as np
import matplotlib.pyplot as plt
from os.path import abspath, dirname, join
import random_walk


# 4b: Perform 1000 random walk for 100 different N = 10,20,30...990,1000.

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
        coords = random_walk.grid_rw_nonreversal(N)  # Generate a random walk
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
fig2.suptitle("Mean squared end-to-end distance of 2D nonreversal random walks")

ax2.errorbar(N_array, MSQ_array, yerr=ERR_array, fmt="xk")
ax2.plot(N_array, p(N_array))

ax2.legend(
    ["Fit:${}+{}x$".format(coeff[1], coeff[0]), "Simulated walk"], loc="upper left"
)

path = join(dirname(abspath(__file__)), "hw04b_msqd_plot_nonreversal.pdf")

fig2.savefig(path)

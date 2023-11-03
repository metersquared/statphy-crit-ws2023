import numpy as np
import matplotlib.pyplot as plt
from os.path import abspath, dirname, join

seed = 42
rng = np.random.default_rng(seed)  # can be called without a seed

n=32768*2

#a: Uniform distribution with values between x=-10 and x=-4.
uniform_dist=rng.uniform(-10,-4,n)

#b: Standard normal distribution with mean 0 and std 1.
normal_dist=rng.standard_normal(n)

#c: Discrete numbers from Poisson distribution with mean 3.
poisson_dist=rng.poisson(3,n)

print(uniform_dist)
print(normal_dist)
print(poisson_dist)


#d: Histogram

plt.rcParams['text.usetex'] = True

# Partition into bins
bins=16

fig, axs = plt.subplots(1, 3, sharey=True, tight_layout=True)

fig.suptitle("Histogram of different distributions")

axs[0].hist(uniform_dist, bins=bins)
axs[1].hist(normal_dist, bins=bins)
axs[2].hist(poisson_dist, bins=bins)

axs[0].set_title("Uniform")
axs[1].set_title("Standard normal")
axs[2].set_title("Poisson")

axs[0].set_xlabel("x")
axs[1].set_xlabel("x")
axs[2].set_xlabel("x")

axs[0].set_ylabel("N")

path = join(dirname(abspath(__file__)),"hw01d_histogram.pdf" )

fig.savefig(path)
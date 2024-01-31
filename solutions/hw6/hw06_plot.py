import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import pandas as pd
import numpy as np
from os.path import abspath, dirname, join

l8 = pd.read_csv("Monte_Carlo_Ising_Importance_L8.csv")
l16 = pd.read_csv("Monte_Carlo_Ising_Importance_L16.csv")
l32 = pd.read_csv("Monte_Carlo_Ising_Importance_L32.csv")

fig, ax = plt.subplots(sharey=True, tight_layout=True)
fig.suptitle("Finite size scaling of $m^2$ in heat-bath Ising model")

ax.errorbar(
    l8["T"].to_numpy(),
    l8["msquared"].to_numpy(),
    yerr=l8["err"].to_numpy(),
    fmt="x",
    label="L=8",
)

ax.errorbar(
    l16["T"].to_numpy(),
    l16["msquared"].to_numpy(),
    yerr=l16["err"].to_numpy(),
    fmt="x",
    label="L=16",
)

ax.errorbar(
    l32["T"].to_numpy(),
    l32["msquared"].to_numpy(),
    yerr=l32["err"].to_numpy(),
    fmt="x",
    label="L=32",
)

ax.legend()
ax.set_ylim((0.0, 1.0))

# Plot vertical line for kBTc.
ax.vlines(2 / np.log(np.sqrt(2) + 1), 0, 1, "k", "dashed")

xt = ax.get_xticks()
xt = np.append(xt, 2 / np.log(np.sqrt(2) + 1))

ax.set_xticks(xt)
ax.xaxis.set_major_formatter(FormatStrFormatter("%.1f"))

path = join(dirname(abspath(__file__)), "hw06_plot.pdf")

fig.savefig(path)
